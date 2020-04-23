# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import cv2
import glob
import os
import tensorflow as tf
from tensorflow.python.keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.layers import Dense
from keras.backend.tensorflow_backend import set_session


def set_gpu_option(which_gpu, fraction_memory):
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = fraction_memory
    config.gpu_options.visible_device_list = which_gpu
    set_session(tf.Session(config=config))

set_gpu_option("0", 0.8)


'''
parameter values  
'''
dataset_name = 'cu50'
model_name = dataset_name
train_dir = '/data/backup/pervinco_2020/pervinco/datasets/' + dataset_name + '/train5'
valid_dir = '//data/backup/pervinco_2020/pervinco/datasets/' + dataset_name + '/valid5'
NUM_CLASSES = len(glob.glob(train_dir + '/*'))

CHANNELS = 3
IMAGE_RESIZE = 224
# EARLY_STOP_PATIENCE must be < NUM_EPOCHS
NUM_EPOCHS = 30
EARLY_STOP_PATIENCE = 3
BATCH_SIZE = 16

saved_path = '/data/backup/pervinco_2020/pervinco/model/'
time = datetime.datetime.now().strftime("%Y.%m.%d_%H:%M") + '_keras'
weight_file_name = '{epoch:02d}-{val_acc:.2f}.hdf5'

if not(os.path.isdir(saved_path + dataset_name + '/' + time)):
    os.makedirs(os.path.join(saved_path + dataset_name + '/' + time))
else:
    pass

'''
train model define
Out intention in this kernel is Transfer Learning by using ResNet50 pre-trained weights except its TOP layer, 
i.e., the xyz_tf_kernels_NOTOP.h5 weights... 
Use this weights as initial weight for training new layer using train images
'''
resnet_weights_path = '/data/backup/pervinco_2020/pervinco/source/weights/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
# resnet_weights_path = '/home/barcelona/다운로드/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

model = Sequential()
model.add(ResNet50(include_top=False, pooling='avg', weights=resnet_weights_path))
model.add(Dense(NUM_CLASSES, activation='sigmoid'))
model.layers[0].trainable = True
model.summary()

optimizer = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# optimizer = optimizers.Adam()
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

'''
image data generator define
'''
data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = data_generator.flow_from_directory(train_dir,
                                                     target_size=(IMAGE_RESIZE, IMAGE_RESIZE),
                                                     batch_size=BATCH_SIZE,
                                                     class_mode='categorical')

validation_generator = data_generator.flow_from_directory(valid_dir,
                                                          target_size=(IMAGE_RESIZE, IMAGE_RESIZE),
                                                          batch_size=BATCH_SIZE,
                                                          class_mode='categorical')

'''
training callbacks define
'''
cb_early_stopper = EarlyStopping(monitor='val_loss', patience=EARLY_STOP_PATIENCE)
cb_checkpointer = ModelCheckpoint(filepath=saved_path + dataset_name + '/' + time + '/' + weight_file_name,
                                  monitor='val_acc', save_best_only=True, mode='auto')

fit_history = model.fit_generator(train_generator,
                                  steps_per_epoch=train_generator.n / BATCH_SIZE,
                                  epochs=NUM_EPOCHS,
                                  validation_data=validation_generator,
                                  validation_steps=validation_generator.n / BATCH_SIZE,
                                  callbacks=[cb_early_stopper, cb_checkpointer])
# train_generator.n / BATCH_SIZE
# validation_generator.n / BATCH_SIZE

model.save(saved_path + dataset_name + '/' + time + '/' + model_name + '.h5')
