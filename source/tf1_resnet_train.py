import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
from tensorflow.python.keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.layers import Dense

'''
parameter values  
'''
NUM_CLASSES = 6
train_dir = '/home/barcelona/pervinco/datasets/face_gender_glass/train'
valid_dir = '/home/barcelona/pervinco/datasets/face_gender_glass/validation'
model_name = 'face_gender_glass'

CHANNELS = 3
IMAGE_RESIZE = 224
# EARLY_STOP_PATIENCE must be < NUM_EPOCHS
NUM_EPOCHS = 50
EARLY_STOP_PATIENCE = 5
BATCH_SIZE_TRAINING = 32
BATCH_SIZE_VALIDATION = 32
saved_path = '/home/barcelona/pervinco/source/weights'

'''
train model define
'''
resnet_weights_path = '/home/barcelona/pervinco/source/weights/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

model = Sequential()
model.add(ResNet50(include_top=False, pooling='avg', weights=resnet_weights_path))
model.add(Dense(NUM_CLASSES, activation='softmax'))
model.layers[0].trainable = True
model.summary()

sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

'''
image data generator define
'''
data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = data_generator.flow_from_directory(
    train_dir,
    target_size=(IMAGE_RESIZE, IMAGE_RESIZE),
    batch_size=BATCH_SIZE_TRAINING,
    class_mode='categorical')

validation_generator = data_generator.flow_from_directory(
    valid_dir,
    target_size=(IMAGE_RESIZE, IMAGE_RESIZE),
    batch_size=BATCH_SIZE_VALIDATION,
    class_mode='categorical')

'''
training callbacks define
'''
cb_early_stopper = EarlyStopping(monitor='val_loss', patience=EARLY_STOP_PATIENCE)
cb_checkpointer = ModelCheckpoint(filepath='./train/product.hdf5', monitor='val_loss',
                                  save_best_only=True, mode='auto')

fit_history = model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.n / BATCH_SIZE_TRAINING,
    epochs=NUM_EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.n / BATCH_SIZE_VALIDATION,
    # callbacks=[cb_early_stopper]
)

model.save(saved_path + model_name + '.h5')