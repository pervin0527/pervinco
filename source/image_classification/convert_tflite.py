import tensorflow as tf

saved_model_dir = "/data/Models/classification/SPC/2022.03.14_14:16"
converter = tf.lite.TFLiteConverter.from_saved_model()

with open("model.tflite", 'wb') as f:
    f.write(tflite_model)