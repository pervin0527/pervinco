import pathlib
import numpy as np
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


def representative_dataset():
  for _ in range(100):
    data = np.random.rand(1, 512, 512, 3)
    yield [data.astype(np.float32)]

label_file_paths='/data/Datasets/Seeds/ETRI_detection/labels.txt'
path = "/home/barcelona/tensorflow/models/research/object_detection/custom/models/fire/21_07_09"
saved_model_dir = f"{path}/saved_model"

tflite_models_dir = pathlib.Path(f'{path}')
tflite_models_dir.mkdir(exist_ok=True, parents=True)

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir, signature_keys=['serving_default'])
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

tflite_fp16_model = converter.convert()
tflite_model_fp16_file = tflite_models_dir/'custom_fp16.tflite'
tflite_model_fp16_file.write_bytes(tflite_fp16_model)