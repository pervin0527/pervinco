# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function
from tqdm import tqdm
from numpy.random import randn

import glob
import pathlib
import random
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from matplotlib.image import imread
import IPython.display as display

AUTOTUNE = tf.data.experimental.AUTOTUNE

# data_dir = tf.keras.utils.get_file('flower_photos','https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz', untar=True)
data_dir = '/home/barcelona/pervinco/datasets/cats_and_dogs_small_set/train'
data_dir = pathlib.Path(data_dir)

all_images = list(data_dir.glob('*/*'))
all_images = [str(path) for path in all_images]
random.shuffle(all_images)

image_count = len(all_images)

all_images[2]

for n in tqdm(all_images[:4]):
    image_path = random.choice(all_images)
    display.display(display.Image(image_path))
    print()

# label_names={'cats': 0, 'dogs': 1}
label_names={}

classes = sorted(glob.glob('/home/barcelona/pervinco/datasets/cats_and_dogs_small_set/train/*'))
idx = 0

for c in classes:
    str_label = c.split('/')[-1]
    label_names.update({str_label : idx})
    idx += 1

print(label_names)


def _process_image(path):

    image = open(path, 'rb').read()

    text=pathlib.Path(filename).parent.name
    label=label_names[text]

    return image,text,label

def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _convert_to_example(image_buffer, label, text):

    example = tf.train.Example(features=tf.train.Features(feature={
        'label': _int64_feature(label),
        'text':  _bytes_feature(tf.compat.as_bytes(text)),
        'encoded': _bytes_feature(tf.compat.as_bytes(image_buffer))}))
    return example

with tf.io.TFRecordWriter('test.tfrecords') as writer:
    for filename in tqdm(all_images):
        image_buffer,text,label = _process_image(filename)
        example = _convert_to_example(image_buffer, label,text)
        writer.write(example.SerializeToString())
