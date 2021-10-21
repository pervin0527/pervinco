# https://www.tensorflow.org/lite/tutorials/model_maker_object_detection
# https://www.tensorflow.org/lite/api_docs/python/tflite_model_maker/config/QuantizationConfig#for_int8
import os
import numpy as np
import tensorflow as tf
import pandas as pd

from absl import logging
from tflite_model_maker.config import ExportFormat
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector
from tflite_model_maker.config import QuantizationConfig

label_file_path = "/data/Datasets/Seeds/DMC/labels/labels.txt"
label_file = pd.read_csv(label_file_path, sep=',', index_col=False, header=None)
label_map = sorted(label_file[0].tolist())
print(label_map)

train_data = object_detector.DataLoader.from_pascal_voc('/data/Datasets/Seeds/DMC/Final/test/images', '/data/Datasets/Seeds/DMC/Final/test/annotations', label_map)
validation_data = object_detector.DataLoader.from_pascal_voc('/data/Datasets/Seeds/DMC/set4/train/images', '/data/Datasets/Seeds/DMC/set4/train/annotations', label_map)

save_path = "/data/Models/efficientdet_lite"
model_file_name = 'efdet_dmc_d0-final-test'

spec = object_detector.EfficientDetLite0Spec(tflite_max_detections=1,
                                             strategy=None,
                                             model_dir=f'{save_path}/{model_file_name}')

model = object_detector.create(train_data,
                               model_spec=spec,
                               epochs=100,
                               batch_size=64,
                               validation_data=validation_data,
                               train_whole_model=True,)

model.export(export_dir=save_path,
             tflite_filename=f'{model_file_name}.tflite',
             label_filename=label_file_path,
             export_format=[ExportFormat.TFLITE, ExportFormat.LABEL])