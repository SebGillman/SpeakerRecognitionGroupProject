import tensorflow as tf
from tflite_runtime.interpreter import Interpreter 
import numpy as np
import time
import os

def decode_audio(audio_binary):
  # Decode WAV-encoded audio files to `float32` tensors, normalized
  # to the [-1.0, 1.0] range. Return `float32` audio and a sample rate.
  audio, _ = tf.audio.decode_wav(contents=audio_binary)
  # Since all the data is single channel (mono), drop the `channels`
  # axis from the array.
  return tf.squeeze(audio, axis=-1)

def get_spectrogram(waveform):
  # Spectogram of a waveform
  # Zero-padding for an audio waveform with less than 16,000 samples.
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
      equal_length, frame_length=255, frame_step=128)
  # Obtain the magnitude of the STFT.
  spectrogram = tf.abs(spectrogram)
  # Add a `channels` dimension, so that the spectrogram can be used
  # as image-like input data with convolution layers (which expect
  # shape (`batch_size`, `height`, `width`, `channels`).
  spectrogram = spectrogram[..., tf.newaxis]
  spectrogram = tf.image.grayscale_to_rgb(spectrogram, name=None)
  return spectrogram

def load_labels(path): 
  # Read the labels from the text file as a Python list.
  with open(path, 'r') as f:
    return [line.strip() for i, line in enumerate(f.readlines())]

def set_input_tensor(interpreter, spectrogram):
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = spectrogram

def classify_audio(interpreter, spectrogram, top_k=1):
  set_input_tensor(interpreter, spectrogram)

  interpreter.invoke()
  output_details = interpreter.get_output_details()[0]
  output = np.squeeze(interpreter.get_tensor(output_details['index']))

  scale, zero_point = output_details['quantization']
  output = scale * (output - zero_point)

  ordered = np.argpartition(-output, 1)
  return [(i, output[i]) for i in ordered[:top_k]][0]

if __name__ == "main":

  model_file = "group_model.tflite"
  label_file = "group_labels.txt"
  output_file = "output.wav"

  interpreter = Interpreter(model_file)
  print("Model Loaded Successfully.")

  interpreter.allocate_tensors()
  _, height, width, _ = interpreter.get_input_details()[0]['shape']
  print("Audio Shape (", width, ",", height, ")")

  # Load a waveform to be classified.
  AUTOTUNE = tf.data.AUTOTUNE
  file = tf.io.gfile.glob(output_file)
  file_ds = tf.data.Dataset.from_tensor_slices(file)
  waveform_ds = decode_audio(file_ds)

  # Convert waveform to Spectrogram
  spectrogram_ds = get_spectrogram(waveform_ds)

  # Classify the audio.
  time1 = time.time()
  label_id, prob = classify_audio(interpreter, spectrogram_ds)
  time2 = time.time()
  classification_time = np.round(time2-time1, 3)
  print("Classificaiton Time =", classification_time, "seconds.")

  # Read class labels.
  labels = load_labels(label_file)

  # Return the classification label of the audio.
  classification_label = labels[label_id]
  print("Audio Label is :", classification_label, ", with Accuracy :", np.round(prob*100, 2), "%.")
  
  # Delete output.wav file
  os.remove(output_file)