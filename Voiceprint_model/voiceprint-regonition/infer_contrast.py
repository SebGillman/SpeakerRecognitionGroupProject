import argparse
import functools
import numpy as np
import tensorflow as tf
from utils.reader import load_audio
from utils.utility import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('audio_path1',      str,    'audio/a_1.wav',          'Path of the first audio file')
add_arg('audio_path2',      str,    'audio/b_2.wav',          'Path of second audio file')
add_arg('input_shape',      str,    '(257, 257, 1)',          'Shape of input data')
add_arg('threshold',        float,   0.7,                     'Threshold of telling if they are from the same person')
add_arg('model_path',       str,    'models/infer_model.h5',  'Path of model')
args = parser.parse_args()

print_arguments(args)

# Load model
model = tf.keras.models.load_model(args.model_path)
model = tf.keras.models.Model(inputs=model.input, outputs=model.get_layer('batch_normalization').output)

# Define input shape of the data
input_shape = eval(args.input_shape)

# Print out the model
model.build(input_shape=input_shape)
model.summary()


# Predict the audio
def infer(audio_path):
    data = load_audio(audio_path, mode='test', spec_len=input_shape[1])
    data = data[np.newaxis, :]
    feature = model.predict(data)
    return feature


if __name__ == '__main__':
    feature1 = infer(args.audio_path1)[0]
    feature2 = infer(args.audio_path2)[0]
    # calculate cos similarity(distance)
    dist = np.dot(feature1, feature2) / (np.linalg.norm(feature1) * np.linalg.norm(feature2))
    if dist > args.threshold:
        print("%s and %s is the same person，similarity is：%f" % (args.audio_path1, args.audio_path2, dist))
    else:
        print("%s and %s is not the same person，similarity is only：%f" % (args.audio_path1, args.audio_path2, dist))
