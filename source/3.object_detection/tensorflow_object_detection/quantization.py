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

label_file_paths='/data/datasets/traffic_sign/labels.txt'
path = "/home/barcelona/tensorflow/models/research/object_detection/custom/models/traffic_sign/21_06_15"
saved_model_dir = f"{path}/saved_model"

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir, signature_keys=['serving_default'])
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
tflite_model_quant = converter.convert()

interpreter = tf.lite.Interpreter(model_content=tflite_model_quant)
input_type = interpreter.get_input_details()[0]['dtype']
print('##### input: ', input_type)
output_type = interpreter.get_output_details()[0]['dtype']
print('##### output: ', output_type)

tflite_model_quant_file = f'{path}/quantize.tflite'
tflite_model_quant_file.write_bytes(tflite_model_quant)