# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import pathlib
import IPython.display as display
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import random

tf.enable_eager_execution()

IMG_HEIGHT, IMG_WIDTH = 192, 192
BATCH_SIZE = 64
EPOCHS = 10
STEPS_PER_EPOCHS = 3

AUTOTUNE = tf.data.experimental.AUTOTUNE
data_root = pathlib.Path('/home/barcelona/pervinco/datasets/face_gender_glass/train')
saved_path = '/home/barcelona/pervinco/exp/weights/'
model_name = 'test_face'
print(data_root)

all_image_paths = list(data_root.glob('*/*'))
all_image_paths = [str(path) for path in all_image_paths]
random.shuffle(all_image_paths)
image_count = len(all_image_paths)
print(image_count)

label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
print('labels : ', label_names, 'label_num : ', len(label_names))
label_to_index = dict((name, index) for index, name in enumerate(label_names))


all_image_labels = [label_to_index[pathlib.Path(path).parent.name]
                    for path in all_image_paths]


def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    image /= 255.0  # normalize to [0,1] range

    return image


def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)


def load_and_preprocess_from_path_label(path, label):
    return load_and_preprocess_image(path), label


def change_range(image, label):
    return 2*image-1, label


path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))

image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))
# print(image_label_ds)

ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_image_labels))

image_label_ds = ds.map(load_and_preprocess_from_path_label)

ds = image_label_ds.shuffle(buffer_size=image_count)
ds = ds.repeat()
ds = ds.batch(BATCH_SIZE)
ds = ds.prefetch(buffer_size=AUTOTUNE)


mobile_net = tf.keras.applications.MobileNetV2(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), include_top=False)
mobile_net.trainable = True

keras_ds = ds.map(change_range)

# The dataset may take a few seconds to start, as it fills its shuffle buffer.
image_batch, label_batch = next(iter(keras_ds))

feature_map_batch = mobile_net(image_batch)
# print(feature_map_batch.shape)


model = tf.keras.Sequential([
  mobile_net,
  tf.keras.layers.GlobalAveragePooling2D(),
  tf.keras.layers.Dense(len(label_names), activation='softmax')])


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=["accuracy"])


# steps_per_epoch = tf.ceil(len(all_image_paths)/BATCH_SIZE).numpy()

model.fit(keras_ds, epochs=EPOCHS, steps_per_epoch=image_count // BATCH_SIZE)

model.save(saved_path + model_name + '.h5')


