import tensorflow as tf
import pathlib

# Quantize model and store it as quantized_tflite_model.tflite in models/
tflite_model = tf.keras.models.load_model('models/infer_model.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(tflite_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model_quant = converter.convert()

tflite_models_dir = pathlib.Path("models/")
tflite_model_quant_file = tflite_models_dir/'infer_quantized_tflite_model.tflite'
tflite_model_quant_file.write_bytes(tflite_model_quant)

