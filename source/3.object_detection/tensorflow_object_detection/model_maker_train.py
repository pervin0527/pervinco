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

label_file_path = "/data/Datasets/Seeds/ETRI_detection/Labels/labels.txt"
label_file = pd.read_csv(label_file_path, sep=',', index_col=False, header=None)
label_map = label_file[0].tolist()
print(label_map)

train_data = object_detector.DataLoader.from_pascal_voc('/data/Datasets/Seeds/ETRI_detection/newset2/train/images', '/data/Datasets/Seeds/ETRI_detection/newset2/train/annotations', label_map)
validation_data = object_detector.DataLoader.from_pascal_voc('/data/Datasets/Seeds/ETRI_detection/newset2/valid/images', '/data/Datasets/Seeds/ETRI_detection/newset2/valid/annotations', label_map)

save_path = "/data/Models/efficientdet_lite"
model_file_name = 'test'

spec = object_detector.EfficientDetLite0Spec(strategy=None,
                                             tflite_max_detections=10,
                                             model_dir=f'{save_path}/{model_file_name}')

model = object_detector.create(train_data,
                               model_spec=spec,
                               epochs=10,
                               batch_size=64,
                               validation_data=validation_data,
                               train_whole_model=True,)

model.export(export_dir=save_path,
             tflite_filename=f'{model_file_name}.tflite',
             label_filename=f'{save_path}/label_map.txt',
             export_format=[ExportFormat.TFLITE, ExportFormat.LABEL])