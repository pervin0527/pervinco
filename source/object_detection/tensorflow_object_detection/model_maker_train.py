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

train_data = "/data/Datasets/SPC/full-name2/test"
valid_data = "/data/Datasets/SPC/full-name2/valid"
label_file_path = "/data/Datasets/SPC/Labels/labels.txt"
save_path = "/data/Models/efficientdet_lite"
model_file_name = "test9"

hparams = {"optimizer" : "sgd",
           "learning_rate" : 0.008,
           "lr_warmup_init" : 0.0008,
           "anchor_scale" : [12.0, 10.0, 8.0, 6.0, 4.0],
           "aspect_ratios" : [8.0, 6.0, 4.0, 2.0, 1.0, 0.5],
           "alpha" : 0.25,
           "gamma" : 2,
           "es" : False,
           "es_monitor" : "val_det_loss",
           "es_patience" : 15
}

label_file = pd.read_csv(label_file_path, sep=',', index_col=False, header=None)
label_map = label_file[0].tolist()
print(label_map)

train_data = object_detector.DataLoader.from_pascal_voc(images_dir=f"{train_data}/images",
                                                        annotations_dir=f"{train_data}/annotations", 
                                                        label_map=label_map, 
)

validation_data = object_detector.DataLoader.from_pascal_voc(images_dir=f'{valid_data}/images',
                                                             annotations_dir=f'{valid_data}/annotations',
                                                             label_map=label_map,
)

spec = object_detector.EfficientDetLite1Spec(hparams=hparams,
                                             verbose=1,
                                             strategy=None, # 'gpus'
                                             tflite_max_detections=1,
                                             model_dir=f'{save_path}/{model_file_name}',
)

model = object_detector.create(train_data,
                               model_spec=spec,
                               epochs=20,
                               batch_size=64,
                               validation_data=validation_data,
                               train_whole_model=True,)

model.export(export_dir=save_path,
             tflite_filename=f'{model_file_name}.tflite',
             label_filename=f'{save_path}/label_map.txt',
             export_format=[ExportFormat.TFLITE])