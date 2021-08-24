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

label_file_path = "/data/Datasets/Seeds/ETRI_detection/custom/labels.txt"
# label_map = ['trafficlight','stop','speedlimit','crosswalk']
# label_map = ['Red_fire_extinguisher', 'Silver_fire_extinguisher', 'fireplug', 'exit_sign', 'fire_detector']
label_file = pd.read_csv(label_file_path, sep=',', index_col=False, header=None)
label_map = sorted(label_file[0].tolist())
print(label_map)

spec = object_detector.EfficientDetLite1Spec()
train_data = object_detector.DataLoader.from_pascal_voc('/data/Datasets/Seeds/ETRI_detection/custom/train/images', '/data/Datasets/Seeds/ETRI_detection/custom/train/annotations', label_map)
validation_data = object_detector.DataLoader.from_pascal_voc('/data/Datasets/Seeds/ETRI_detection/custom/valid/images', '/data/Datasets/Seeds/ETRI_detection/custom/valid/annotations', label_map)

model = object_detector.create(train_data,
                               model_spec=spec,
                               epochs=100,
                               batch_size=16,
                               train_whole_model=True,
                               validation_data=validation_data)

# model.evaluate(validation_data)

model.export(export_dir='/data/Models/efficientdet_lite',
             tflite_filename='efdet_d1_etri.tflite',
             label_filename=label_file_path,
             export_format=[ExportFormat.TFLITE, ExportFormat.LABEL])