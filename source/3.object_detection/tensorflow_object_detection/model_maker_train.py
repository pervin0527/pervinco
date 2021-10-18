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
label_map = {1:"giant", 2:"notgiant"}
print(label_map)

train_data = object_detector.DataLoader.from_pascal_voc('/data/Datasets/Seeds/DMC/set5/train/images', '/data/Datasets/Seeds/DMC/set5/train/annotations', label_map)
validation_data = object_detector.DataLoader.from_pascal_voc('/data/Datasets/Seeds/DMC/set5/valid/images', '/data/Datasets/Seeds/DMC/set5/valid/annotations', label_map)

save_path = "/data/Models/efficientdet_lite"
# model_file_name = 'efdet_dmc_d1_set4'

spec = object_detector.EfficientDetLite1Spec(tflite_max_detections=1,
                                             strategy=None,
                                             model_dir=f'{save_path}/{model_file_name}')
model = object_detector.create(train_data,
                               model_spec=spec,
                               epochs=100,
                               batch_size=64,
                               train_whole_model=True,
                               validation_data=validation_data)

# config = QuantizationConfig.for_float16()
# config = QuantizationConfig.for_int8(representative_data=validation_data,
#                                      quantization_steps=10, 
#                                      inference_input_type=tf.int8, 
#                                      inference_output_type=tf.int8, 
#                                      supported_ops=tf.lite.OpsSet.TFLITE_BUILTINS_INT8)

model.export(export_dir=save_path,
             tflite_filename=f'{model_file_name}.tflite',
            #  label_filename=label_file_path,
            #  saved_model_filename="saved_model",
            #  quantization_config=config,
            #  export_format=[ExportFormat.TFLITE, # ExportFormat.SAVED_MODEL, # ExportFormat.LABEL])
             export_format=[ExportFormat.TFLITE])

# model.evaluate_tflite(f'{save_path}/{model_file_name}.tflite', validation_data)