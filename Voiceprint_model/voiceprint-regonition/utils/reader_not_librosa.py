import random

import tensorflow as tf
import librosa
import numpy as np


# Load and pre-process audio file
def load_audio_not_librosa(audio_path, mode='train', win_length=400, sr=16000, hop_length=160, n_fft=512, spec_len=257):
    # Load audio
    # Decode WAV-encoded audio files to `float32` tensors, normalized
  # to the [-1.0, 1.0] range. Return `float32` audio and a sample rate.
    audio, _ = tf.audio.decode_wav(contents=tf.io.read_file(audio_path))
    # Since all the data is single channel (mono), drop the `channels`
    # axis from the array.
    tf.squeeze(audio, axis=-1)

    input_len = 16000
    waveform = waveform[:input_len]
    zero_padding = tf.zeros(
      [16000] - tf.shape(waveform),
      dtype=tf.float32)
    # Cast the waveform tensors' dtype to float32.
    waveform = tf.cast(waveform, dtype=tf.float32)
    # Concatenate the waveform with `zero_padding`, which ensures all audio
    # clips are of the same length.
    equal_length = tf.concat([waveform, zero_padding], 0)
    # Convert the waveform to a spectrogram via a STFT.
    spectrogram = tf.signal.stft(
        equal_length, frame_length=win_length, frame_step=hop_length)
    # Obtain the magnitude of the STFT.
    spectrogram = tf.abs(spectrogram)
    # Add a `channels` dimension, so that the spectrogram can be used
    # as image-like input data with convolution layers (which expect
    # shape (`batch_size`, `height`, `width`, `channels`).
    spectrogram = spectrogram[..., tf.newaxis]
    spectrogram = tf.image.grayscale_to_rgb(spectrogram, name=None)
    return spectrogram


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
