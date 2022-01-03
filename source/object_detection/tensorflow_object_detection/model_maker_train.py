# https://www.tensorflow.org/lite/tutorials/model_maker_object_detection
# https://www.tensorflow.org/lite/api_docs/python/tflite_model_maker/config/QuantizationConfig#for_int8
# https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/third_party/efficientdet/hparams_config.py
# tensorflow_examples/lite/model_maker/third_party/efficientdet/keras/train_lib.py

import os
import numpy as np
import tensorflow as tf
import pandas as pd

from absl import logging
from tflite_model_maker.config import ExportFormat
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector
from tflite_model_maker.config import QuantizationConfig

train_data = "/data/Datasets/SPC/full-name5/train"
valid_data = "/data/Datasets/SPC/full-name2/valid"
label_file_path = "/data/Datasets/SPC/Labels/labels.txt"
save_path = "/data/Models/efficientdet_lite"
model_file_name = "test2"

label_file = pd.read_csv(label_file_path, sep=',', index_col=False, header=None)
label_map = label_file[0].tolist()
print(label_map)

train_data = object_detector.DataLoader.from_pascal_voc(images_dir=f"{train_data}/images",
                                                        annotations_dir=f"{train_data}/annotations", 
                                                        label_map=label_map, 
                                                        # num_shards=1,
                                                        # cache_dir=f"{save_path}/{model_file_name}/data"
)

validation_data = object_detector.DataLoader.from_pascal_voc(images_dir=f'{valid_data}/images',
                                                             annotations_dir=f'{valid_data}/annotations',
                                                             label_map=label_map,
                                                            #  num_shards=1,
                                                            #  cache_dir=f"{save_path}/{model_file_name}/data"
)

spec = object_detector.EfficientDetLite1Spec(strategy=None, # 'gpus'
                                             tflite_max_detections=10,
                                             model_dir=f'{save_path}/{model_file_name}',
                                             verbose=1)

model = object_detector.create(train_data,
                               model_spec=spec,
                               epochs=10,
                               batch_size=64,
                               validation_data=validation_data,
                               train_whole_model=True,)

model.export(export_dir=save_path,
             tflite_filename=f'{model_file_name}.tflite',
             label_filename=f'{save_path}/label_map.txt',
             export_format=[ExportFormat.TFLITE])


"""
/tensorflow_examples/lite/model_maker/core/task/object_detector.py
/tensorflow_examples/lite/model_maker/third_party/efficientdet/keras/efficientdet_keras.py
/tensorflow_examples/lite/model_maker/third_party/efficientdet/keras/postprocess.py
/tensorflow_examples/lite/model_maker/core/task/custom_model.py
"""