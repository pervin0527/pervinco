from __future__ import absolute_import, division, print_function, unicode_literals
import os
import cv2
import tensorflow as tf
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import datetime
from tensorflow import keras


tf.executing_eagerly()
AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 128
IMG_HEIGHT = 227
IMG_WIDTH = 227
epochs = 1200

valid_dir = pathlib.Path('/home/barcelona/pervinco/datasets/cats_and_dogs_filtered/validation')
total_val_data = len(list(valid_dir.glob('*/*.jpg')))
print(total_val_data)
CLASS_NAMES = np.array([item.name for item in valid_dir.glob('*') if item.name != "LICENSE.txt"])

model = tf.keras.models.load_model('/home/barcelona/pervinco/model/good/'
                                   'max99_2class_2019.12.20_16:03:04/ALEX1_2class_2019.12.20_16:03:04.h5')
# optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, decay=5e-5, momentum=0.9)
# model.compile(
#     # optimizer='adam',
#     optimizer=optimizer,
#     # loss='binary_crossentropy'
#     loss='categorical_crossentropy',
#     metrics=['accuracy']
# )

valid_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

valid_generator = valid_image_generator.flow_from_directory(
    directory=valid_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)


history = model.evaluate_generator(
    valid_generator,
    steps=total_val_data//BATCH_SIZE,
    verbose = 1
)