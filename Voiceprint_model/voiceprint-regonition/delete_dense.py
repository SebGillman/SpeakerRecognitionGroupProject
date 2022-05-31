import argparse
import functools
import os
import shutil

import numpy as np
import tensorflow as tf

from utils.utility import add_arguments, print_arguments

from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization, Dense


parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('model_path',       str,    'D:/P/VoiceprintRecognition-Tensorflow-develop/models/infer_model.h5',  'path to old model')
add_arg('save_model_path',  str,    'new_models/',                'new model path')
args = parser.parse_args()

print_arguments(args)

# load model
model = tf.keras.models.load_model(args.model_path)
model = tf.keras.models.Model(inputs=model.input, outputs=model.get_layer('batch_normalization').output)

new_model = tf.keras.Sequential()
#new_model = Model(inputs=model.inputs)
#for layer in model.layers[:-1]:  # ignore this, wrong method 
#    new_model.add(layer)
new_model = model
new_model.summary()

new_model.save('my_model.h5')
