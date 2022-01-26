import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
import pandas as pd

from absl import logging
from tflite_model_maker.config import ExportFormat
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector
from tflite_model_maker.config import QuantizationConfig

train_data = "/data/Datasets/SPC/test/train"
valid_data = "/data/Datasets/SPC/test/valid"
label_file_path = "/data/Datasets/SPC/Labels/labels.txt"
save_path = "/data/Models/efficientdet_lite"
model_file_name = "test-90"

hparams = {"optimizer" : "sgd",
           "learning_rate" : 0.008,
           "lr_warmup_init" : 0.0008,
           "anchor_scale" : 7.0,
           "aspect_ratios" : [8.0, 4.0, 2.0, 1.0, 0.5],
           "num_scales" : 5,
           "alpha" : 0.25,
           "gamma" : 2,
           "es" : False,
           "es_monitor" : "val_det_loss",
           "es_patience" : 15,
           "ckpt" : None}

label_file = pd.read_csv(label_file_path, sep=',', index_col=False, header=None)
label_map = label_file[0].tolist()
print(label_map)

train_data = object_detector.DataLoader.from_pascal_voc(images_dir=f"{train_data}/images",
                                                        annotations_dir=f"{train_data}/annotations", 
                                                        label_map=label_map)

validation_data = object_detector.DataLoader.from_pascal_voc(images_dir=f'{valid_data}/images',
                                                             annotations_dir=f'{valid_data}/annotations',
                                                             label_map=label_map)

spec = object_detector.EfficientDetLite1Spec(verbose=1,
                                             strategy=None, # 'gpus'
                                             hparams=hparams,
                                             tflite_max_detections=10,
                                             model_dir=f'{save_path}/{model_file_name}')

model = object_detector.create(train_data,
                               model_spec=spec,
                               epochs=90,
                               batch_size=64,
                               validation_data=validation_data,
                               train_whole_model=True,)

model.export(export_dir=save_path,
             tflite_filename=f'{model_file_name}.tflite',
             label_filename=f'{save_path}/label_map.txt',
             export_format=[ExportFormat.TFLITE])