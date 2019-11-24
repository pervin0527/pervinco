# -*- coding: utf-8 -*-
# Reference - https://www.tensorflow.org/guide/data#top_of_page
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import pathlib
import tensorflow as tf
import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Layer

tf.executing_eagerly()

'''
definition dataset
consuming sets of files - https://www.tensorflow.org/guide/data#consuming_sets_of_files
'''
AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 1500
IMG_HEIGHT = 224
IMG_WIDTH = 224

train_ds_dir = pathlib.Path('/home/barcelona/pervinco/datasets/flower_photos')
train_data_num = len(list(train_ds_dir.glob('*/*.jpg')))
print(train_data_num)

# train_dataset에서 jpg files를 list 형태로 반환.
train_data_list = tf.data.Dataset.list_files(str(train_ds_dir/'*/*'))

CLASS_NAMES = np.array([item.name for item in train_ds_dir.glob('*') if item.name != "LICENSE.txt"])
# print(CLASS_NAMES)


# list 내용을 확인 sample 10개
# for f in train_data_list.take(10):
#     print(f.numpy())


'''
define functions
'''


def get_label(file_path):
    # convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    print(parts)
    # The second to last is the class-directory
    # return parts[-2] == CLASS_NAMES
    return parts[-2]


def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])


def process_path(file_path):
    label = get_label(file_path)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label


def one_hot_encoding(labels):
    encoder = LabelEncoder()
    encoder.fit(labels)
    encoded = encoder.transform(labels)
    return encoded


def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000):
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()

    ds = ds.shuffle(buffer_size=shuffle_buffer_size)
    ds = ds.repeat()
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds


mapping = train_data_list.map(process_path, num_parallel_calls=AUTOTUNE)
# print(mapping)
#
# for image, label in mapping.take(1):
#   print("Image shape: ", image.numpy().shape)
#   print("Label: ", label.numpy())

train_data = prepare_for_training(mapping)
train_images, train_labels = next(iter(train_data))
train_labels = one_hot_encoding(train_labels)
# print(train_images)
# print(train_labels)
print(train_images.shpae)
print(train_labels.shape)

'''
define ALEX NET
ALEX_NET keras ver - https://datascienceschool.net/view-notebook/d19e803640094f76b93f11b850b920a4/
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

# optimizer = optimizers.SGD(lr=0.01, decay=5e-5, momentum=0.9)
# model.compile(optimizer=optimizer,
#               # loss='binary_crossentropy'
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])


model.compile(optimizer='adam',
              # loss='binary_crossentropy'
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

result = model.fit(train_images, train_labels, epochs=100)

test_image = cv2.imread('/home/barcelona/pervinco/datasets/predict/test_roses.jpg', cv2.IMREAD_COLOR)
test_image = cv2.resize(test_image, (224, 224))
test_image = tf.dtypes.cast(test_image, dtype=tf.float32)
test_image = tf.reshape(test_image, [1, 224, 224, 3])
predictions = model.predict(test_image)

predict_label = np.argmax(predictions[0])
print(CLASS_NAMES[predict_label])


