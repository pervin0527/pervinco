from __future__ import absolute_import, division, print_function, unicode_literals
import os
import pathlib
from tensorflow.keras.models import Sequentail
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

'''
definition dataset
'''

train_dataset_dir = pathlib.Path('/home/barcelona/pervinco/datasets/flower_photos')
train_data_num = len(list(train_dataset_dir.glob('*/*.jpg')))
print(train_data_num)