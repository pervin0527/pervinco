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
    # Conv2D(16, 3, padding='same', activation='relu',
    #        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    # MaxPooling2D(),
    # Dropout(0.2),
    # Conv2D(32, 3, padding='same', activation='relu'),
    # MaxPooling2D(),
    # Conv2D(64, 3, padding='same', activation='relu'),
    # MaxPooling2D(),
    # Dropout(0.2),
    # Flatten(),
    # Dense(512, activation='relu'),
    # Dense(5, activation='softmax')
    Conv2D(filters=96, kernel_size=(11, 11), strides=4, padding='same', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    Conv2D(filters=256, kernel_size=(5, 5), padding='same', activation='relu'),
    # BatchNormalization(),
    MaxPooling2D(pool_size=(3, 3), strides=2),
    Conv2D(filters=384, kernel_size=(3, 3), padding='same', activation='relu'),
    # BatchNormalization(),
    MaxPooling2D(pool_size=(3, 3), strides=2),
    Conv2D(filters=384, kernel_size=(3, 3), padding='same', activation='relu'),
    Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'),
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

model.load_weights('/home/barcelona/pervinco/model/ALEX_NET_TRAIN_MODEL.h5')

# eval_image = cv2.imread('/home/barcelona/pervinco/datasets/predict/test_sunflower2.jpg')
# eval_image = cv2.resize(eval_image, (224, 224))
# eval_image = tf.dtypes.cast(eval_image, dtype=tf.float32)
# eval_image = tf.reshape(eval_image, [1, 224, 224, 3])
#
# predictions = model.predict(eval_image)
# result = np.argmax(predictions[0])
# print('Categori : ', CLASS_NAMES)
# print('predict label number :', result)
# print('predict result is : ', CLASS_NAMES[result])

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