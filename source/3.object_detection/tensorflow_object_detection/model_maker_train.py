import os
import numpy as np
import tensorflow as tf
import pandas as pd

from absl import logging
from tflite_model_maker.config import ExportFormat
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector

assert tf.__version__.startswith('2')
tf.get_logger().setLevel('ERROR')
logging.set_verbosity(logging.ERROR)

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

label_file_path = "/data/Datasets/Seeds/DMC/labels/labels.txt"
label_file = pd.read_csv(label_file_path, sep=',', index_col=False, header=None)
label_map = sorted(label_file[0].tolist())
print(label_map)


save_path = "/data/Models/efficientdet_lite"
model_file_name = 'efdet_dmc_d1_set4-2000'

# spec = object_detector.EfficientDetLite0Spec(model_dir=save_path)
spec = object_detector.EfficientDetLite1Spec()
train_data = object_detector.DataLoader.from_pascal_voc('/data/Datasets/Seeds/DMC/set4/train/images', '/data/Datasets/Seeds/DMC/set4/train/annotations', label_map)
validation_data = object_detector.DataLoader.from_pascal_voc('/data/Datasets/Seeds/DMC/set4/valid/images', '/data/Datasets/Seeds/DMC/set4/valid/annotations', label_map)

model = object_detector.create(train_data,
                               model_spec=spec,
                               epochs=2000,
                               batch_size=64,
                               train_whole_model=True,
                               validation_data=validation_data)

# print(model.evaluate(validation_data))

model.export(export_dir=save_path,
             tflite_filename=f'{model_file_name}.tflite',
            #  saved_model_filename = "saved_model",
             label_filename=label_file_path,
             export_format=[ExportFormat.TFLITE,
                            # ExportFormat.SAVED_MODEL,
                            ExportFormat.LABEL])