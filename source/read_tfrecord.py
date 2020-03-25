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
image_dataset = tf.data.TFRecordDataset('test.tfrecords')

IMG_SIZE=224
# Create a dictionary describing the features.  
image_feature_description = {
    'label': tf.io.FixedLenFeature([], tf.int64),
    'text': tf.io.FixedLenFeature([], tf.string),  
    'encoded': tf.io.FixedLenFeature([], tf.string),
}

def _parse_image_function(example_proto):
    # Parse the input tf.Example proto using the dictionary above.
    feature=tf.io.parse_single_example(example_proto, image_feature_description)

    image=feature['encoded']
    image = tf.image.decode_jpeg(image, channels=3)
    #   image = tf.image.resize_images(image, [224, 224])
    #   image /= 255.0  # normalize to [0,1] range

    image = tf.cast(image, tf.float32)
    image = (image/127.5) - 1
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))

    return image,feature['label']

dataset = image_dataset.map(_parse_image_function)

BATCH_SIZE = 32

ds = dataset.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=1000))
ds = dataset.shuffle(buffer_size=1000)
ds = ds.repeat()
ds = ds.batch(BATCH_SIZE)
ds = ds.prefetch(buffer_size=AUTOTUNE)
