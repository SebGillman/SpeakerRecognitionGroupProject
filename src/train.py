import argparse
import functools
import os
import shutil
import time
from datetime import datetime, timedelta

import tensorflow as tf
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization, Dense

from utils import reader
from utils.loss import ArcLoss
from utils.metrics import ArcNet
from utils.utility import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('gpus',             str,    '0',                      'The GPU used when training, ignore it if you plan to train on colab. format as：0,1')
add_arg('batch_size',       int,    16,                       'batch size')
add_arg('num_epoch',        int,    50,                       'epoch')
add_arg('num_classes',      int,    3242,                     'number of labels')
add_arg('learning_rate',    float,  1e-3,                     'learning rate')
add_arg('input_shape',      str,    '(128, 257, 1)',          'shape of input data')
add_arg('train_list_path',  str,    'dataset/train_list.txt', 'path to train list')
add_arg('test_list_path',   str,    'dataset/test_list.txt',  'path to test list')
add_arg('save_model_path',  str,    'models/',                'path where you want to save the model')
add_arg('pretrained_model', str,    'models/model_weights.h5','path of the pre-trained model, type in None if no pretrained model or when to start from scratch')
args = parser.parse_args()


# Save model
def save_model(model):
    if not os.path.exists(args.save_model_path):
        os.makedirs(args.save_model_path)
    infer_model = Model(inputs=model.input, outputs=model.get_layer('feature_output').output)
    infer_model.save(filepath=os.path.join(args.save_model_path, 'infer_model.h5'), include_optimizer=False)
    model.save_weights(filepath=os.path.join(args.save_model_path, 'model_weights.h5'))

def create_model(input_shape):
    # Obtain model
    model = tf.keras.Sequential()
    model.add(ResNet50V2(input_shape=input_shape, include_top=False, weights=None, pooling='max'))
    model.add(BatchNormalization())
    model.add(Dense(units=512, kernel_regularizer=tf.keras.regularizers.l2(5e-4), name='feature_output'))
    model.add(ArcNet(num_classes=args.num_classes))
    return model


# Train
def main():
    shutil.rmtree('log', ignore_errors=True)
    input_shape = eval(args.input_shape)
    # Below code used is for training using muliple Gpus, change it if you want to run code in colab
    strategy = tf.distribute.MirroredStrategy()
    BATCH_SIZE = args.batch_size * strategy.num_replicas_in_sync

    with open(args.train_list_path, 'r') as f:
        lines = f.readlines()
    epoch_step_sum = int(len(lines) / BATCH_SIZE)

    # Obtain traing and test results
    train_dataset = reader.train_reader(data_list_path=args.train_list_path,
                                        batch_size=BATCH_SIZE,
                                        num_epoch=args.num_epoch,
                                        spec_len=input_shape[1])
    test_dataset = reader.test_reader(data_list_path=args.test_list_path,
                                      batch_size=BATCH_SIZE,
                                      spec_len=input_shape[1])
    # Support training on multiple Gpus
    train_dataset = strategy.experimental_distribute_dataset(train_dataset)
    test_dataset = strategy.experimental_distribute_dataset(test_dataset)

    with strategy.scope():
        model = create_model(input_shape)
        # print model
        model.build(input_shape=input_shape)
        model.summary()
        # define optimazation method
        boundaries = [10 * i * epoch_step_sum for i in range(1, args.num_epoch // 10, 1)]
        lr = [0.1 ** l * args.learning_rate for l in range(len(boundaries) + 1)]
        scheduler = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=boundaries, values=lr)
        optimizer = tf.keras.optimizers.SGD(learning_rate=scheduler, momentum=0.9)

    with strategy.scope():
        # Load pre-trained model
        if args.pretrained_model is not None and os.path.exists(args.pretrained_model):
            model.load_weights(args.pretrained_model, by_name=True, skip_mismatch=True)
            print('Load pre-trained model successfully!')

    with strategy.scope():
        train_loss_metrics = tf.keras.metrics.Mean(name='train_loss')
        train_accuracy_metrics = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        test_loss_metrics = tf.keras.metrics.Mean(name='test_loss')
        test_accuracy_metrics = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

        train_loss_metrics.reset_states()
        train_accuracy_metrics.reset_states()

        train_summary_writer = tf.summary.create_file_writer('log/train')
        test_summary_writer = tf.summary.create_file_writer('log/test')

    with strategy.scope():
        # Define loss
        loss_object = ArcLoss(num_classes=args.num_classes, reduction=tf.keras.losses.Reduction.NONE)

        def compute_loss(label, prediction):
            per_example_loss = loss_object(prediction, label)
            return tf.nn.compute_average_loss(per_example_loss, global_batch_size=BATCH_SIZE)

    with strategy.scope():
        def train_step(inputs):
            sounds, labels = inputs
            # start training
            with tf.GradientTape() as tape:
                predictions = model(sounds)
                # obtain loss
                train_loss = compute_loss(labels, predictions)

            # update gradient
            gradients = tape.gradient(train_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            #calculate average loss and accuracy
            train_loss_metrics(train_loss)
            train_accuracy_metrics(labels, predictions)
            return train_loss

        def test_step(inputs):
            sounds, labels = inputs
            # start evaluation
            predictions = model(sounds)
            # obtain loss
            test_loss = compute_loss(labels, predictions)
            #calculate average loss and accuracy
            test_loss_metrics(test_loss)
            test_accuracy_metrics(labels, predictions)

    with strategy.scope():
        # Original code use `experimental_run_v2` as it is using tensorflow 2.3, my version is 2.4 so directly use strategy.run
        #This function can copy the calculation and run it distributly
        @tf.function
        def distributed_train_step(dataset_inputs):
            per_replica_losses = strategy.run(train_step, args=(dataset_inputs,))
            return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

        @tf.function
        def distributed_test_step(dataset_inputs):
            return strategy.run(test_step, args=(dataset_inputs,))

        # Start training
        train_step_num = 0
        test_step_num = 0
        count_step = epoch_step_sum * args.num_epoch
        start = time.time()
        for step, train_inputs in enumerate(train_dataset):
            distributed_train_step(train_inputs)

            # output in the log
            if step % 100 == 0:
                eta_sec = ((time.time() - start) * 1000) * (count_step - step)
                eta_str = str(timedelta(seconds=int(eta_sec / 1000)))
                print("[%s] Step [%d/%d], Loss %f, Accuracy %f, Learning rate %f, eta: %s" % (
                    datetime.now(), step, count_step, train_loss_metrics.result(), train_accuracy_metrics.result(),
                    optimizer._decayed_lr('float32').numpy(), eta_str))

                # record data
                with train_summary_writer.as_default():
                    tf.summary.scalar('Loss', train_loss_metrics.result(), step=train_step_num)
                    tf.summary.scalar('Accuracy', train_accuracy_metrics.result(), step=train_step_num)
                train_step_num += 1
            # evaluate the model
            if step % epoch_step_sum == 0 and step != 0:
                for test_inputs in test_dataset:
                    distributed_test_step(test_inputs)
                print('=================================================')
                print("[%s] Test Loss %f, Accuracy %f" % (datetime.now(), test_loss_metrics.result(), test_accuracy_metrics.result()))
                print('=================================================')
                # record data
                with test_summary_writer.as_default():
                    tf.summary.scalar('Loss', test_loss_metrics.result(), step=test_step_num)
                    tf.summary.scalar('Accuracy', test_accuracy_metrics.result(), step=test_step_num)
                test_step_num += 1
                test_loss_metrics.reset_states()
                test_accuracy_metrics.reset_states()

                # save model
                save_model(model)
            start = time.time()


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    print_arguments(args)
    main()
