# -*- coding: utf-8 -*-

'''
import
'''
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
AUTOTUNE = tf.data.experimental.AUTOTUNE
import IPython.display as display
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
tf.executing_eagerly()
'''
define param, path
'''
print(tf.__version__)
print(tf.test.is_gpu_available())
IMG_HEIGHT, IMG_WIDTH = 192, 192
BATCH_SIZE = 32
steps_per_epoch = 10

data_dir = '/home/barcelona/pervinco/datasets/cats_and_dogs_small_set/train'
data_dir = pathlib.Path(data_dir)
image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)

CLASS_NAMES = sorted(np.array([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"]))
print(CLASS_NAMES)


def get_label(file_path):
    # convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    # The second to last is the class-directory
    print(parts)
    return parts[-2] == CLASS_NAMES


def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    img = tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])
    # normalize to [0, 1] range
    img /= 255.0
    # normalize to [-1, 1] range
    img = 2*img-1
    return img


def process_path(file_path):
    label = get_label(file_path)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label


def prepare_for_training(ds):
    ds = ds.shuffle(buffer_size=image_count)
    ds = ds.repeat()
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds


list_ds = tf.data.Dataset.list_files(str(data_dir / '*/*'))

# 데이터셋 입력 체크
for f in list_ds.take(5):
    print(f.numpy())

labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)

for image, label in labeled_ds.take(1):
    print("Image shape: ", image.numpy().shape)
    print("Label: ", label.numpy())

train_ds = prepare_for_training(labeled_ds)
image_batch, label_batch = next(iter(train_ds))

mobile_net = tf.keras.applications.MobileNetV2(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), include_top=False)
mobile_net.trainable = False

feature_map_batch = mobile_net(image_batch)
print(feature_map_batch.shape)

model = tf.keras.Sequential([
    mobile_net,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(len(CLASS_NAMES), activation='softmax')])

model.summary()

model.fit(image_batch, epochs=1, steps_per_epoch=3)


