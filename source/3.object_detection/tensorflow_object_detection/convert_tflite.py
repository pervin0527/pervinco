import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from tflite_support.metadata_writers import object_detector
from tflite_support.metadata_writers import writer_utils

label_file_paths='/data/datasets/traffic_sign/labels.txt'
path = "/home/barcelona/tensorflow/models/research/object_detection/custom/models/traffic_sign/21_06_14"
saved_model_dir = f"{path}/saved_model"

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir, signature_keys=['serving_default'])
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.experimental_new_converter = True
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = converter.convert()

with tf.io.gfile.GFile(f'{path}/custom.tflite', 'wb') as f:
  f.write(tflite_model)

writer = object_detector.MetadataWriter.create_for_inference(writer_utils.load_file(f'{path}/custom.tflite'), input_norm_mean=[0], input_norm_std=[255], label_file_paths=[label_file_paths])
writer_utils.save_file(writer.populate(), f'{path}/custom.tflite')