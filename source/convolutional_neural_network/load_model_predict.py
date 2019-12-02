# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals
import os
import cv2
import tensorflow as tf
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import glob
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

train_dir = pathlib.Path('/home/barcelona/pervinco/datasets/cats_and_dogs_filtered/train')
CLASS_NAMES = sorted(np.array([item.name for item in train_dir.glob('*') if item.name != "LICENSE.txt"]))


tf.executing_eagerly()
AUTOTUNE = tf.data.experimental.AUTOTUNE
IMG_HEIGHT = 224
IMG_WIDTH = 224

def ALEX_NET():
    inputs = keras.Input(shape=(224, 224, 3))

    conv1 = keras.layers.Conv2D(filters=96, kernel_size=(11, 11), strides=4, padding='same',
                                input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
                                activation='relu')(inputs)

    conv2 = keras.layers.Conv2D(filters=256, kernel_size=(5, 5), padding='same', activation='relu')(conv1)
    norm1 = tf.nn.local_response_normalization(conv2)
    pool1 = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2)(norm1)

    conv3 = keras.layers.Conv2D(filters=384, kernel_size=(3, 3), padding='same', activation='relu')(pool1)
    norm2 = tf.nn.local_response_normalization(conv3)
    pool2 = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2)(norm2)

    conv4 = keras.layers.Conv2D(filters=384, kernel_size=(3, 3), padding='same', activation='relu')(pool2)
    conv5 = keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(conv4)
    pool3 = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2)(conv5)

    flat = keras.layers.Flatten()(pool3)
    dense1 = keras.layers.Dense(4096, activation='relu')(flat)
    drop1 = keras.layers.Dropout(0.5)(dense1)
    dense2 = keras.layers.Dense(4096, activation='relu')(drop1)
    drop2 = keras.layers.Dropout(0.5)(dense2)
    dense3 = keras.layers.Dense(2, activation='softmax')(drop2)
    return keras.Model(inputs=inputs, outputs=dense3)



model = ALEX_NET()
model.summary()

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, decay=5e-5, momentum=0.9)
model.compile(
    # optimizer='adam',
    optimizer=optimizer,
    # loss='binary_crossentropy'
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

'''
@breif
저장된 모델을 불러오는 load_weights
https://www.tensorflow.org/tutorials/keras/save_and_load#top_of_page
이때, train 된 모델을 그대로 위에 선언 되어 있어야 weight file이 load 될 수 있다.
ex) training을 ALEX_NET으로 했다면 model.Sequentail에 ALEX_NET model이 구축되어 있어야 함.
'''
model.load_weights('/home/barcelona/pervinco/model/ALEX_cat_dog.h5')

eval_dir = glob.glob('/home/barcelona/pervinco/datasets/predict/cat_dog/*.jpg')
# print(eval_dir)
print('Categori : ', CLASS_NAMES)
for img in eval_dir:
    print('input image : ', img)
    img = cv2.imread(img)
    img = cv2.resize(img, (224, 224))
    img = tf.dtypes.cast(img, dtype=tf.float32)
    img = tf.reshape(img, [1, 224, 224, 3])

    predictions = model.predict(img)
    print(predictions)
    result = np.argmax(predictions[0])
    # print('predict label number :', result)
    print(CLASS_NAMES[result])