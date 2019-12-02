# -*- coding: utf-8 -*-
'''
@author 김민준
@log
            ---------------------------------default ver-------------------------------------------
            epochs : 1000, loss: 0.2951, accuracy: 0.8958, predict result : False
            ---------------------------------update 19.11.26 --------------------------------------
            modify test data input format
            epochs : 2000, loss: 0.2216, accuracy: 0.92 predict result : 7개중 4개 정답.
            ---------------------------------update 19.11.28---------------------------------------
            training model change ALEX_NET ---> tutorial model
            added Early Stopping, Tensorboard
            epochs : 256/500 (early stopping) accuracy: 0.85 predict result : 7개중 2개 정답.
            ---------------------------------update 19.11.29---------------------------------------
            training model change Tutorial_model ------> ALEX_NET
            epochs : 217/500 (early stopping) accuracy : 0.96 loss : 0.1131

'''
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import cv2
import tensorflow as tf
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import datetime
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.utils import np_utils

tf.executing_eagerly()
AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 128
IMG_HEIGHT = 224
IMG_WIDTH = 224
epochs = 300

'''
@brief - 데이터셋 다운로드 링크
Dogs vs cats dataset : https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip
flower dataset : https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz

training dataset 경로 설정
'''
train_dir = pathlib.Path('/home/barcelona/AutoCrawler/dataset')
valid_dir = pathlib.Path('/home/barcelona/pervinco/datasets/flower_photos')
total_train_data = len(list(train_dir.glob('*/*.jpg')))
total_val_data = len(list(valid_dir.glob('*/*.jpg')))
print(total_train_data)
CLASS_NAMES = np.array([item.name for item in train_dir.glob('*') if item.name != "LICENSE.txt"])

'''
@brief
model 설계
ALEX_NET keras ver - https://datascienceschool.net/view-notebook/d19e803640094f76b93f11b850b920a4/

tensorflow API
Conv2D - https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D
MaxPool2D - https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool2D
'''
model = Sequential([
    Input(shape=(224, 224, 3)),
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
    # # Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
    # Dense(5, activation='softmax')

    ##  ▲tutorial model▲  ▼ALEXNET▼

    Conv2D(filters=96, kernel_size=(11, 11), strides=4, padding='same', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
           activation='relu'),

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


# optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, decay=5e-5, momentum=0.9)
model.compile(
    optimizer='adam',
    # optimizer=optimizer,
    # loss='binary_crossentropy'
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

'''
@brief
train ImageDataGenerator를 이용해 data preprocessing 적용
rescale : pixel value를 0~1사이 값으로 설정하기 위함.
rotation_range : image를 회전
width_shift_range : 지정된 수평 방향 이동 범위 내에서 임의로 원본 이미지를 이동. 좌우로 움직이는 것으로 이해하면 쉬움
height_shift_range : 지정된 수직 방향 이동 범위 내에서 임의로 원본 이미지를 이동. 상하로 움직이는 것으로 이해하면 쉬움.
horizontal_flip : 수평 방향으로 뒤집기. 상하 반전을 적용하는 것으로 이해하면 쉬움.
zoom_range : 지정된 확대/축소 범위로 원본 이미지를 확대/축소

자세한 설명은 해당 링크 참조. https://tykimos.github.io/2017/06/10/CNN_Data_Augmentation/
'''
# train_image_generator = ImageDataGenerator(rescale=1./255)
train_image_generator = ImageDataGenerator(rescale=1./255,
                                           rotation_range=45,
                                           width_shift_range=.15,
                                           height_shift_range=.15,
                                           horizontal_flip=True,
                                           zoom_range=0.5,
                                           # shear_range=0.2
                                           )

valid_image_generator = ImageDataGenerator(rescale=1./255)
'''
@brief
아래 API에서 flow_from_directory 부분 참조.
target_size = 모든 이미지의 크기를 재조정할 치수.
class_mode = dataset과 분류 문제인지 회귀 문제인지에 따라 설정.
    (categorical : 2D형태의 one-hot encoding 된 label
     binary : 1D형태의 이진 label
     sparse : 1D형태의 정수 label)
batch_size = 데이터 배치 크기
shuffle = train일 경우 True, valid일 경우 False
https://keras.io/ko/preprocessing/image/
'''
train_generator = train_image_generator.flow_from_directory(
    directory=train_dir,
    # resize train data
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    shuffle=True,
    class_mode='categorical',
)

valid_generator = valid_image_generator.flow_from_directory(
    directory=valid_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)
'''
@brief
callback 정의 + Tensorboard를 이용하기 위한 log에 대한 정의.
'''
log_dir = '/home/barcelona/pervinco/model/logs' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

history = model.fit_generator(
    train_generator,
    steps_per_epoch=total_train_data//BATCH_SIZE,
    epochs=epochs,
    verbose=1,
    validation_data=valid_generator,
    validation_steps=total_val_data//BATCH_SIZE,
    callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, verbose=1), tensorboard_callback]
)

# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']
#
# loss = history.history['loss']
# val_loss = history.history['val_loss']
#
# epochs_range = range(epochs)
#
# plt.figure(figsize=(8, 8))
# plt.subplot(1, 2, 1)
# plt.plot(epochs_range, acc, label='Training Accuracy')
# plt.plot(epochs_range, val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')
#
# plt.subplot(1, 2, 2)
# plt.plot(epochs_range, loss, label='Training Loss')
# plt.plot(epochs_range, val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss')
# plt.show()

model.save('/home/barcelona/pervinco/model/ALEX_reg_train.h5')
