from PIL import Image
import numpy as np
import boto3
import os
import logging
from botocore.exceptions import ClientError


def stft_to_jpeg(spec_mag, label, bucket_name='stft-data'):
    # Take a spectrogram and save as jpeg in s3 bucket
    if len(spec_mag.shape) > 2:
        spec_mag_1 = spec_mag[:, :, 0]
    im = Image.fromarray(spec_mag_1)

    file_name = label+'jpeg'
    im.save(file_name)

    if object_name is None:
        object_name = os.path.basename(file_name)

    # Upload the file
    s3_client = boto3.client('s3')
    try:
        response = s3_client.upload_file(file_name, bucket_name, object_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True

