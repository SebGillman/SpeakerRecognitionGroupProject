import argparse
import functools
import os
import shutil
import time
import sys
import shutil

import numpy as np

import tensorflow as tf

from utils.reader import load_audio
from utils.record import RecordAudio
from utils.utility import add_arguments, print_arguments
from AWS.s3_upload_file import upload_file
from AWS.s3_download_file import download_files

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('audio_db',         str,    'audio_db',               'path to our audio database')
add_arg('input_shape',      str,    '(257, 257, 1)',          'shape of input data')
add_arg('threshold',        float,   0.75,                     'threshold of verification')
add_arg('model_path',       str,    'models/infer_model.h5',  'path to model')
args = parser.parse_args()

print_arguments(args)


# Load model
model = tf.keras.models.load_model(args.model_path)
model = tf.keras.models.Model(inputs=model.input, outputs=model.get_layer('batch_normalization').output)

# obtain average
input_shape = eval(args.input_shape)

# print out the model
model.build(input_shape=input_shape)
model.summary()

person_feature = []
person_name = []

# Cloud metadata
wav_bucket_name = 'armgroupproject'
stft_bucket_name = 'stft-data'
unlabelled_stft_bucket_name = 'unlabelled-stft-data'

def infer(audio_path, message = False, stft_cloud=False, name=None, mode='infer'):
    time5 = time.time()
    data = load_audio(audio_path, mode=mode, spec_len=input_shape[1], name=name, stft_cloud=stft_cloud)
    time6 = time.time()
    stft_time = np.round(time6-time5, 3)

    data = data[np.newaxis, :]

    time3 = time.time()
    feature = model.predict(data)
    time4 = time.time()
    prediction_time = np.round(time4-time3)

    if message:
        print('STFT time: {} seconds.'.format(stft_time))
        print("Prediction time = {} seconds.".format(prediction_time))

    return feature

def load_audio_db(audio_db_path, message = False):
    audios = os.listdir(audio_db_path)
    message = False
    for audio in audios:
        path = os.path.join(audio_db_path, audio)
        name = audio[:-4]
        feature = infer(path, message, mode='load')[0]
        person_name.append(name)
        person_feature.append(feature)
        if message:
            print("Loaded %s audio." % name)


# Voicprint recognition
def recognition(path, mode='unlabelled', cloud_db=False):
    name = ''
    pro = 0
    feature = infer(path, mode=mode, stft_cloud=cloud_db)[0]
    for i, person_f in enumerate(person_feature):
        dist = np.dot(feature, person_f) / (np.linalg.norm(feature) * np.linalg.norm(person_f))
        if dist > pro:
            pro = dist
            name = person_name[i]

    return name, pro


# Register new member
def register(path, user_name, cloud_db=False):
    save_path = os.path.join(args.audio_db, user_name + os.path.basename(path)[-4:])
    shutil.move(path, save_path)
    message = False
    feature = infer(save_path, message, name=user_name, stft_cloud=cloud_db)[0]
    person_name.append(user_name)
    person_feature.append(feature)

    if cloud_db:
        wav_success_upload = upload_file(save_path, wav_bucket_name)
        if wav_success_upload:
             print('Successfully uploaded audio: {} to the cloud!'.format(user_name+'.wav'))

             # delete file from 'audio_db' after upload --> other solution: save recording to different directory and then depending on cloud_db either join it with 'audio_db' or 'tmp'
             #os.remove('tmp/'+user_name+'.wav') # removes file from the local database
    else:
        print('Successfully saved audio: {} to the local database!'.format(user_name+'.wav'))


if __name__ == '__main__':
    load_audio_db(args.audio_db, message = True)
    record_audio = RecordAudio()

    flag = False

    print('\n \n \n')
    
    try:
        while True:
            print('\n-----------------------------------------------------------------------------------------------------')
            select_fun = input("Please type in number to choose function:\n type in 0 to register new member,\n type in 1 to do single voice recognition,\n type in 2 to do continuous voice recognition, \n type in 3 to exit the program. \n")

            if select_fun == '0':
                audio_path = record_audio.record()
                name = input("Please type in your name as new member: ")
                if name == '': continue
                while True:
                    cloud_db=input('\nPlease type 1 if you want to store your audio to the cloud, else type 0 to store it in the local database\n')
                    if cloud_db in ['0','1']:
                        break
                    else:
                        print('Not correct input')
                cloud_db = bool(int(cloud_db))
                register(audio_path, name, cloud_db)

            elif select_fun == '1':
                # download 
                while True:
                    cloud_db=input('\nPlease type 1 if you want to access the cloud database, else type 0 to access the local database\n')
                    if cloud_db in ['0','1']:
                        break
                    else:
                        print('Not correct input')
                cloud_db = bool(int(cloud_db))
                
                if cloud_db and not flag:
                    time_1 = time.time()
                    print('Accessing Cloud Database...')
                    wav_download = download_files(wav_bucket_name)
                    load_audio_db("tmp")
                    time_2 = time.time()
                    flag = True
                    print('Download time = ', np.round(time_2-time_1, 3), ' seconds.')

                # run inference 
                audio_path = record_audio.record(cloud=cloud_db)
                time1 = time.time()
                name, p = recognition(audio_path, mode='unlabelled', cloud_db=cloud_db)
                time2 = time.time()
                if p > args.threshold and "Noise" not in name:
                    print("The one currently speaking is %s with a similarity of %f" % (name.split("_")[0], p))
                    print('Classification time = ', np.round(time2-time1, 3), ' seconds. \n')
                else:
                    print("There's no matched member in the database,try speaking in your natural tone or avoid noisy enviroment \n")

                if not cloud_db:
                    os.remove('audio_db/temp.wav')

            elif select_fun == '2':
                # download 
                while True:
                    cloud_db=input('\nPlease type 1 if you want to access the cloud database, else type 0 to access the local database\n')
                    if cloud_db in ['0','1']:
                        break
                    else:
                        print('Not correct input')
                cloud_db = bool(int(cloud_db))

                if cloud_db and not flag:
                    time_1 = time.time()
                    print("Downloading database...")
                    wav_download = download_files(wav_bucket_name)
                    load_audio_db("tmp")
                    time_2 = time.time()
                    flag = True
                    print('Download time = ', np.round(time_2-time_1, 3), ' seconds.')

                # run inference 
                print("\nRecording has started, press Ctrl+C to quit")
                print("[RECORDER] Listening ...... \n")
                keypress=False
                try:
                    while True:
                        audio_path = record_audio.recordconst(cloud=cloud_db)
                        time1 = time.time()
                        name, p = recognition(audio_path, mode='unlabelled', cloud_db=cloud_db)
                        time2 = time.time()
                        if p > args.threshold and "Noise" not in name:
                            print("The one currently speaking is %s with a similarity of %f" % (name.split("_")[0], p))
                            print('Classification time = ', np.round(time2-time1, 3), ' seconds. \n')
                        else:
                            print("There's no matched member in the database,try speaking in your natural tone or avoid noisy enviroment \n")
                except KeyboardInterrupt:
                    if not cloud_db:
                        os.remove('audio_db/temp.wav')
                    pass

            elif select_fun == '3':
                print('Exiting program...')
                if os.path.exists('./tmp'):
                    shutil.rmtree('./tmp')
                else:
                    pass
                sys.exit()

            else:
                print('\nPlease type either 0, 1, 2 or 3 \n')
                
    except KeyboardInterrupt:
        pass
