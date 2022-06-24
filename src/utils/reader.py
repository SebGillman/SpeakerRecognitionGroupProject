import random

import tensorflow as tf
import librosa
import numpy as np
from librosa.display import specshow
import matplotlib.pyplot as plt
import boto3
import os
import logging
from botocore.exceptions import ClientError

# Load and pre-process audio file
def load_audio(audio_path, mode='train', win_length=400, sr=16000, hop_length=160, n_fft=512, spec_len=257, object_name=None, stft_cloud=False, name=None):
    # Load audio
    wav, sr_ret = librosa.load(audio_path, sr=sr)
    if mode == 'train':
        extended_wav = np.append(wav, wav)
        if np.random.random() < 0.3:
            extended_wav = extended_wav[::-1]
    else:
        extended_wav = np.append(wav, wav[::-1])
    # calculate STFT
    linear = librosa.stft(extended_wav, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
    mag, _ = librosa.magphase(linear)

    
    # save the STFT in folder speactrograms
    if mode != 'train' and stft_cloud == False:
        destination = '/home/pi/SpeakerRecognitionGroupProject/src/spectrograms/'

        if name is not None:
            file_name = name+'.png'
        else:
            png_name = audio_path.replace('audio_db/', '')
            file_name = png_name+'.png'
        
        plt.figure()
        librosa.display.specshow(mag, sr=sr, hop_length=hop_length, y_axis='log', x_axis='time')

        try:
            plt.savefig(os.path.join(destination + file_name))
            success = True
        except:
            print("error")
            success = False

        if success:
            print('Saved STFT: {} in the local folder!'.format(file_name))

    if stft_cloud and mode != 'load':
        if name is not None:
            file_name = name+'.png'
        elif mode == 'unlabelled':
            file_name = str(np.random.randint(1000))+'.png'
        else:
            file_name = audio_path+'.png'

        plt.figure()
        librosa.display.specshow(mag, sr=sr, hop_length=hop_length, y_axis='log', x_axis='time')
        plt.savefig(file_name)

        if object_name is None:
            object_name = os.path.basename(file_name)

        # Upload the file
        s3_client = boto3.client('s3')
        try:
            if mode == 'unlabelled':
                bucket_name = 'unlabelled-stft-data'
            elif mode == 'infer':
                bucket_name = 'stft-data'
            response = s3_client.upload_file(file_name, bucket_name, object_name)
            #destination = os.path.join('./spectrograms', file_name)
        except ClientError as e:
            logging.error(e)
            success = False
        success = True

        if success:
            print('Uploaded STFT: {} to the cloud!'.format(file_name))

        if os.path.exists(file_name):
            os.remove(file_name)
        else:
            print(file_name)

    freq, freq_time = mag.shape
    assert freq_time >= spec_len, "speaking time should be greater than 1.3s"
    if mode == 'train':
        # random cut
        rand_time = np.random.randint(0, freq_time - spec_len)
        spec_mag = mag[:, rand_time:rand_time + spec_len]
    else:
        spec_mag = mag[:, :spec_len]
    mean = np.mean(spec_mag, 0, keepdims=True)
    std = np.std(spec_mag, 0, keepdims=True)
    spec_mag = (spec_mag - mean) / (std + 1e-5)
    spec_mag = spec_mag[:, :, np.newaxis]

    return spec_mag


# preprocess the  data
def data_generator(data_list_path, spec_len=257):
    with open(data_list_path, 'r') as f:
        lines = f.readlines()
        random.shuffle(lines)
    for line in lines:
        audio_path, label = line.replace('\n', '').split('\t')
        spec_mag = load_audio(audio_path, mode='train', spec_len=spec_len)
        yield spec_mag, np.array(int(label))


# Load training data
def train_reader(data_list_path, batch_size, num_epoch, spec_len=257):
    ds = tf.data.Dataset.from_generator(generator=lambda:data_generator(data_list_path, spec_len=spec_len),
                                        output_types=(tf.float32, tf.int64))

    train_dataset = ds.shuffle(buffer_size=1000) \
        .batch(batch_size=batch_size) \
        .repeat(num_epoch) \
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return train_dataset


# Load test data
def test_reader(data_list_path, batch_size, spec_len=257):
    ds = tf.data.Dataset.from_generator(generator=lambda:data_generator(data_list_path, spec_len=spec_len),
                                        output_types=(tf.float32, tf.int64))

    test_dataset = ds.batch(batch_size=batch_size)
    return test_dataset
