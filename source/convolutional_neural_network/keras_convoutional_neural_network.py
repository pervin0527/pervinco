# -*- coding: utf-8 -*-
'''
@author 김민준
@HACK accuracy가 높지 않음. predict 결과가 정확하지 못함.
'''
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
BATCH_SIZE = 128
IMG_HEIGHT = 224
IMG_WIDTH = 224
epochs = 250

'''
@brief - 데이터셋 다운로드 링크
Dogs vs cats dataset : https://www.kaggle.com/c/dogs-vs-cats/data
flower dataset : https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz

training dataset 경로 설정
'''
train_dir = pathlib.Path('/home/barcelona/pervinco/datasets/flower_photos')
total_train_data = len(list(train_dir.glob('*/*.jpg')))
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
                                           zoom_range=0.5)

'''
@brief
아래 API에서 flow_from_directory 부분 참조.
target_size = 모든 이미지의 크기를 재조정할 치수.
class_mode = dataset과 분류 문제인지 회귀 문제인지에 따라 설정.
    (categorical : 2D형태의 one-hot encoding된 label
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
    class_mode='sparse',
)

history = model.fit_generator(
    train_generator,
    steps_per_epoch=total_train_data//BATCH_SIZE,
    epochs=epochs
)

'''
@brief
학습된 모델에 test 이미지를 넣고, 해당 이미지의 class를 정확하게 맞추는지 검증.
'''
test_image = cv2.imread('/home/barcelona/pervinco/datasets/predict/ddlion_test.jpg', cv2.IMREAD_COLOR)
test_image = cv2.resize(test_image, (224, 224))
test_image = tf.dtypes.cast(test_image, dtype=tf.float32)
test_image = tf.reshape(test_image, [1, 224, 224, 3])
predictions = model.predict(test_image)

result = np.argmax(predictions[0])
print('predict result is : ', CLASS_NAMES[result])