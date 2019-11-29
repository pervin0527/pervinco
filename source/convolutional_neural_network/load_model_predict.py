# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals
import os
import cv2
import tensorflow as tf
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import glob
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

train_dir = pathlib.Path('/home/barcelona/AutoCrawler/dataset')
CLASS_NAMES = np.array([item.name for item in train_dir.glob('*') if item.name != "LICENSE.txt"])


tf.executing_eagerly()
AUTOTUNE = tf.data.experimental.AUTOTUNE
IMG_HEIGHT = 224
IMG_WIDTH = 224

model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu',
           input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D(),
    Dropout(0.2),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Dropout(0.2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(5, activation='softmax')
    ##  ▲tutorial model  ▼ALEXNET
    # Conv2D(filters=96, kernel_size=(11, 11), strides=4, padding='same', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    # Conv2D(filters=256, kernel_size=(5, 5), padding='same', activation='relu'),
    # # BatchNormalization(),
    # MaxPooling2D(pool_size=(3, 3), strides=2),
    # Conv2D(filters=384, kernel_size=(3, 3), padding='same', activation='relu'),
    # # BatchNormalization(),
    # MaxPooling2D(pool_size=(3, 3), strides=2),
    # Conv2D(filters=384, kernel_size=(3, 3), padding='same', activation='relu'),
    # Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'),
    # MaxPooling2D(pool_size=(3, 3), strides=2),
    # Flatten(),
    # Dense(4096, activation='relu'),
    # Dropout(0.5),
    # Dense(4096, activation='relu'),
    # Dropout(0.5),
    # Dense(5, activation='softmax')
])

model.compile(optimizer='adam',
              # loss='binary_crossentropy'
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

'''
@breif
저장된 모델을 불러오는 load_weights
https://www.tensorflow.org/tutorials/keras/save_and_load#top_of_page
이때, train 된 모델을 그대로 위에 선언 되어 있어야 weight file이 load 될 수 있다.
ex) training을 ALEX_NET으로 했다면 model.Sequentail에 ALEX_NET model이 구축되어 있어야 함.
'''
model.load_weights('/home/barcelona/pervinco/model/tutorial_network_flower.h5')

eval_dir = glob.glob('/home/barcelona/pervinco/datasets/predict/*.jpg')
# print(eval_dir)
print('Categori : ', CLASS_NAMES)
for img in eval_dir:
    print('input image : ', img)
    img = cv2.imread(img)
    img = cv2.resize(img, (224, 224))
    img = tf.dtypes.cast(img, dtype=tf.float32)
    img = tf.reshape(img, [1, 224, 224, 3])

    predictions = model.predict(img)
    result = np.argmax(predictions[0])
    # print('predict label number :', result)
    print('predict result is : ', CLASS_NAMES[result])