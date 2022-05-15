import pyarmnn as ann
import numpy as np
import tensorflow as tf

print('Working with Arm NN version ' + ann.ARMNN_VERSION)

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

if __name__ == "main":

    #Load a waveform
    AUTOTUNE = tf.data.AUTOTUNE
    file = tf.io.gfile.glob("output.wav")
    file_ds = tf.data.Dataset.from_tensor_slices(file)
    waveform_ds = decode_audio(file_ds)

    # Convert waveform to Spectrogram
    spectrogram_ds = get_spectrogram(waveform_ds)


    # ONNX, Caffe and TF parsers also exist.
    parser = ann.ITfLiteParser()
    network = parser.CreateNetworkFromBinaryFile("./group_model.tflite")

    graph_id = 0
    input_names = parser.GetSubgraphInputTensorNames(graph_id)
    input_binding_info = parser.GetNetworkInputBindingInfo(graph_id, input_names[0])
    input_tensor_id = input_binding_info[0]
    input_tensor_info = input_binding_info[1]
    print('tensor id: ' + str(input_tensor_id))
    print('tensor info: ' + str(input_tensor_info))
    # Create a runtime object that will perform inference.
    options = ann.CreationOptions()
    runtime = ann.IRuntime(options)

    # Backend choices earlier in the list have higher preference.
    preferredBackends = [ann.BackendId('CpuAcc'), ann.BackendId('CpuRef')]
    opt_network, messages = ann.Optimize(network, preferredBackends, runtime.GetDeviceSpec(), ann.OptimizerOptions())

    # Load the optimized network into the runtime.
    net_id, _ = runtime.LoadNetwork(opt_network)
    print("Loaded network, id={net_id}")
    # Create an inputTensor for inference.
    input_tensors = ann.make_input_tensors([input_binding_info], [spectrogram_ds])

    # Get output binding information for an output layer by using the layer name.
    output_names = parser.GetSubgraphOutputTensorNames(graph_id)
    output_binding_info = []
    for i in output_names:
        output_binding_info.append(parser.GetNetworkOutputBindingInfo(0, output_names[0]))
    output_tensors = ann.make_output_tensors(output_binding_info)

    runtime.EnqueueWorkload(0, input_tensors, output_tensors)
    results = ann.workload_tensors_to_ndarray(output_tensors)
    print(results[0])
    label_id = np.argmax(results)

    # Read class labels.
    labels = load_labels("group_labels.txt")

    classification_label = labels[label_id]
    print("Audio Label is :", classification_label)

