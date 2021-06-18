import tensorflow as tf
import numpy as np
import os

from tflite_model_maker.config import ExportFormat
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector

spec = model_spec.get('efficientdet_lite0')

train_data = object_detector.DataLoader.from_csv("/data/datasets/traffic_sign/train.csv")
model = object_detector.create(train_data, model_spec=spec, batch_size=8, train_whole_model=True)