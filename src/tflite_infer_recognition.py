import argparse
import functools
import os
import shutil
import time
import sys

import numpy as np
import tensorflow as tf

from tflite_runtime.interpreter import Interpreter

from utils.reader import load_audio
from utils.record import RecordAudio
from utils.utility import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('audio_db',         str,    'audio_db',               'path to our audio database')
add_arg('input_shape',      str,    '(257, 257, 1)',          'shape of input data')
add_arg('threshold',        float,   0.7,                     'threshold of verification')
add_arg('model_path',       str,    'models/infer_quantized_tflite_model.tflite',  'path to model')
args = parser.parse_args()

print_arguments(args)

# Load Model
interpreter = Interpreter(args.model_path)
interpreter.allocate_tensors()
print("TFLite Quantized Model Loaded Successfully.")

interpreter.allocate_tensors()
_, height, width, _ = interpreter.get_input_details()[0]['shape']
print("Audio Shape (", width, ",", height, ")")

# obtain average
input_shape = eval(args.input_shape)

person_feature = []
person_name = []


# predict the audio
def infer(audio_path):
    time5 = time.time()
    data = load_audio(audio_path, mode='infer', spec_len=input_shape[1])
    time6 = time.time()
    stft_time = np.round(time6-time5, 3)
    print('STFT time: {} seconds.'.format(stft_time))

    data_tf = tf.convert_to_tensor(data[np.newaxis, :])

    time3 = time.time()
    output_details = interpreter.get_output_details()[0]
    input_details = interpreter.get_input_details()[0]
    #print('input details: {}'.format(input_details))
    #print('output details: {}'.format(output_details))

    interpreter.set_tensor(input_details['index'], data_tf)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details['index'])
    time4 = time.time()
    predict_time = np.round(time4-time3, 3)
    print("Prediction time = {} seconds.".format(predict_time))
    return output


# Load the database and print out the list of members
def load_audio_db(audio_db_path):
    audios = os.listdir(audio_db_path)
    for audio in audios:
        path = os.path.join(audio_db_path, audio)
        name = audio[:-4]
        feature = infer(path)[0]
        person_name.append(name)
        person_feature.append(feature)
        print("Loaded %s audio." % name)


# Voicprint recognition
def recognition(path):
    name = ''
    pro = 0
    feature = infer(path)[0]
    for i, person_f in enumerate(person_feature):
        dist = np.dot(feature, person_f) / (np.linalg.norm(feature) * np.linalg.norm(person_f))
        if dist > pro:
            pro = dist
            name = person_name[i]
    return name, pro


# Register new member
def register(path, user_name):
    save_path = os.path.join(args.audio_db, user_name + os.path.basename(path)[-4:])
    shutil.move(path, save_path)
    feature = infer(save_path)[0]
    person_name.append(user_name)
    person_feature.append(feature)


if __name__ == '__main__':
    load_audio_db(args.audio_db)
    record_audio = RecordAudio()

    print('\n \n \n')

    try:
        while True:
            print('\n------------------------------------------------------------------')
            select_fun = int(input("Please type in number to choose function:\n type in 0 to register new member,\n type in 1 to do voice recognition,\n type in 2 to do continuous recognition, \n type in 3 to exit the program. \n"))
            if select_fun == 0:
                audio_path = record_audio.record()
                name = input("Please type in your name as new member: ")
                if name == '': continue
                register(audio_path, name)
            elif select_fun == 1:
                audio_path = record_audio.record()
                time1 = time.time()
                name, p = recognition(audio_path)
                time2 = time.time()
                if p > args.threshold:
                    print("The one currently speaking is %s with a similarity of %f" % (name, p))
                    print('Classification time = ', np.round(time2-time1, 3), ' seconds.')
                else:
                    print("There's no matched member in the database,try speaking in your natural tone or avoid noisy enviroment")
            elif select_fun == 2:
                print("\nRecording has started, press Ctrl+C to quit")
                print("[RECORDER] Listening ...... \n")
                keypress=False
                try:
                    while True:
                        audio_path = record_audio.recordconst()
                        time1 = time.time()
                        name, p = recognition(audio_path)
                        time2 = time.time()
                        if p > args.threshold:
                            print("The one currently speaking is %s with a similarity of %f" % (name, p))
                            print('Classification time = ', np.round(time2-time1, 3), ' seconds. \n')
                        else:
                            print("There's no matched member in the database,try speaking in your natural tone or avoid noisy enviroment \n")
                except KeyboardInterrupt:
                    pass
            elif(select_fun==3):
                print('Exiting program...')
                sys.exit()
            else:
                print('Please type either 0, 1, 2 or 3 \n')
                
    except KeyboardInterrupt:
        pass
