import os
import numpy as np
import tensorflow as tf

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

label_map = ['trafficlight','stop','speedlimit','crosswalk']

spec = object_detector.EfficientDetLite0Spec()
train_data = object_detector.DataLoader.from_pascal_voc('/data/Datasets/Seeds/mm_data/train', '/data/Datasets/Seeds/mm_data/train', label_map)
validation_data = object_detector.DataLoader.from_pascal_voc('/data/Datasets/Seeds/mm_data/test', '/data/Datasets/Seeds/mm_data/test', label_map)

model = object_detector.create(train_data,
                               model_spec=spec,
                               epochs=500,
                               batch_size=20,
                               train_whole_model=True,
                               validation_data=validation_data)

model.evaluate(validation_data)

model.export(export_dir='/data/Models/efficientdet_lite',
             tflite_filename='efficientdet-lite-d0.tflite',
             label_filename='/data/Datasets/Seeds/traffic_sign/labels.txt',
             export_format=[ExportFormat.TFLITE, ExportFormat.LABEL])