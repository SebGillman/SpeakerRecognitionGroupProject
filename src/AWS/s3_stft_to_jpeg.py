import numpy as np
import boto3
import os
import logging
from botocore.exceptions import ClientError
import skimage.io
from PIL import Image


def PIL_stft_to_jpeg(spec_mag, label, bucket_name='stft-data', object_name=None):
    # Take a spectrogram and save as jpeg in s3 bucket
    if len(spec_mag.shape) > 2:
        spec_mag_1 = spec_mag[:, :, 0]
    im = Image.fromarray(spec_mag_1)

    if im.mode != 'RGB':
        im = im.convert('RGB')

    file_name = label+'.jpeg'
    im.save(file_name)

    if object_name is None:
        object_name = os.path.basename(file_name)

    # Upload the file
    s3_client = boto3.client('s3')
    try:
        response = s3_client.upload_file(file_name, bucket_name, object_name)
    except ClientError as e:
        logging.error(e)
        success = False
    success = True

    if os.path.exists(file_name):
        os.remove(file_name)
    else:
        print(file_name)

    return success

def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled

def stft_to_jpeg(spec_mag, label, bucket_name='stft-data', object_name=None):
    # Take a spectrogram and save as jpeg in s3 bucket
    # min-max scale to fit inside 8-bit range
    img = scale_minmax(spec_mag, 0, 255).astype(np.uint8)
    img = np.flip(img, axis=0) # put low frequencies at the bottom in image
    img = 255-img # invert. make black==more energy

    file_name = label+'.jpeg'
    # save as PNG
    skimage.io.imsave(file_name, img)

    if object_name is None:
        object_name = os.path.basename(file_name)

    # Upload the file
    s3_client = boto3.client('s3')
    try:
        response = s3_client.upload_file(file_name, bucket_name, object_name)
    except ClientError as e:
        logging.error(e)
        success = False
    success = True

    if os.path.exists(file_name):
        os.remove(file_name)
    else:
        print(file_name)

    return success


    

