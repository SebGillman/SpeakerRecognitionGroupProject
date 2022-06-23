import numpy as np
import os
from PIL import Image


def stft_to_jpeg(spec_mag, label, output_path='./spectrograms'):
    # Take a spectrogram and save as jpeg in output path 
    if len(spec_mag.shape) > 2:
        spec_mag_1 = spec_mag[:, :, 0]
    im = Image.fromarray(spec_mag_1)

    if im.mode != 'RGB':
        im = im.convert('RGB')

    file_name = label+'.jpeg'
    im.save(file_name)

    destination = os.path.join(output_path, file_name)
    success = True

    """
    if os.path.exists(output_path + "/" + file_name):
        os.remove(file_name)
    else:
        print(file_name)
    """
    return success