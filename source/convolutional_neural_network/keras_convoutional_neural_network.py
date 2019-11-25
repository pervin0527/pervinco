# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import cv2
import tensorflow as tf
import pathlib
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

tf.executing_eagerly()
AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 200
IMG_HEIGHT = 224
IMG_WIDTH = 224
epochs = 30

train_dir = pathlib.Path('/home/barcelona/pervinco/datasets/flower_photos')
total_train_data = len(list(train_dir.glob('*/*.jpg')))
print(total_train_data)
CLASS_NAMES = np.array([item.name for item in train_dir.glob('*') if item.name != "LICENSE.txt"])


model = Sequential([
    Conv2D(filters=96, kernel_size=(11, 11), strides=4, padding='same', activation='relu',
           input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D(pool_size=(3, 3), strides=2),
    Conv2D(filters=256, kernel_size=(5, 5), strides=1, padding='same', activation='relu'),
    MaxPooling2D(pool_size=(3, 3), strides=2),
    Conv2D(filters=384, kernel_size=(3, 3), strides=1, padding='same', activation='relu'),
    Conv2D(filters=384, kernel_size=(3, 3), strides=1, padding='same', activation='relu'),
    Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same', activation='relu'),
    MaxPooling2D(pool_size=(3, 3), strides=2),
    Flatten(),
    Dense(4096, activation='relu'),
    Dropout(0.5),
    Dense(4096, activation='relu'),
    Dropout(0.5),
    Dense(5, activation='softmax')
])

model.compile(optimizer='adam',
              # loss='binary_crossentropy'
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

train_data_gen = ImageDataGenerator(rescale=1./255)
train_generator = train_data_gen.flow_from_directory(
    directory=train_dir,
    # resize train data
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    # binary crossentropy loss func
    # class_mode='binary'
    class_mode='binary'
)

history = model.fit_generator(
    train_generator,
    steps_per_epoch=total_train_data//BATCH_SIZE,
    epochs=epochs
)

test_image = cv2.imread('/home/barcelona/pervinco/predict/ddlion_test.jpg', cv2.IMREAD_COLOR)
test_image = cv2.resize(test_image, (224, 224))
test_image = tf.dtypes.cast(test_image, dtype=tf.float32)
test_image = tf.reshape(test_image, [1, 224, 224, 3])
predictions = model.predict(test_image)

result = np.argmax(predictions[0])
print(CLASS_NAMES[result])