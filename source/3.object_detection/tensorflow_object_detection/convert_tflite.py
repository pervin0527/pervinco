import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from tflite_support.metadata_writers import object_detector
from tflite_support.metadata_writers import writer_utils

# GPU setup
gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 1:
    try:
        print("Activate Multi GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
    except RuntimeError as e:
        print(e)

else:
    try:
        print("Activate Sigle GPU")
        tf.config.experimental.set_memory_growth(gpus[0], True)
        strategy = tf.distribute.experimental.CentralStorageStrategy()
    except RuntimeError as e:
        print(e)


label_file_paths='/data/datasets/traffic_sign/labels.txt'
path = "/home/barcelona/tensorflow/models/research/object_detection/custom/models/traffic_sign/21_06_15"
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