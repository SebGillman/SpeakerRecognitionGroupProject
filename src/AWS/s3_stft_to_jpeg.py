from PIL import Image
import numpy as np
from AWS.s3_upload_file import upload_file

def stft_to_jpeg(spec_mag, label, bucket_name='stft-data'):
    # Take a spectrogram and save as jpeg in s3 bucket
    im = Image.fromarray(spec_mag)

    filename = label+'jpeg'
    im.save(filename)

    success_upload = upload_file(filename, bucket_name)
    return success_upload

