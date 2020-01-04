# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras import datasets, layers, models
import numpy as np
import pathlib
from sklearn.preprocessing import LabelEncoder

tf.executing_eagerly()
AUTOTUNE = tf.data.experimental.AUTOTUNE

train_data_dir = pathlib.Path('/home/barcelona/pervinco/datasets/flower_photos')
train_data_num = len(list(train_data_dir.glob('*/*.jpg')))
print(train_data_num)

CLASS_NAMES = np.array([item.name for item in train_data_dir.glob('*')])
print(CLASS_NAMES)

list_ds = tf.data.Dataset.list_files(str(train_data_dir / '*/*'))

BATCH_SIZE = 300
IMG_HEIGHT = 150
IMG_WIDTH = 150
STEPS_PER_EPOCH = np.ceil(train_data_num / BATCH_SIZE)

def get_label(file_path):
    # convert the path to a list of path components
    parts = tf.strings.split([file_path], '/')
    # print(CLASS_NAMES)
    return parts.values[-2]


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


labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)

'''
for image, label in labeled_ds.take(1):
    print("Image shape: ", image.numpy().shape)
    label = label.numpy().decode('utf-8') == CLASS_NAMES
    label = one_hot_encoding(label)
    print("Label: ", label)
'''

train_ds = prepare_for_training(labeled_ds)
train_image, train_label = next(iter(train_ds))
train_label = one_hot_encoding(train_label)
train_image = train_image / 255.0
# print(train_image)
# print(train_label)
print(train_image.shape, train_label.shape)

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(5, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_image, train_label, epochs=40)