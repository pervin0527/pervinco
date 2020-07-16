"""
Usage:
  # From tensorflow/models/
  # Create train data:
  python generate_tfrecord.py --csv_input=data/train_labels.csv  --output_path=train.record
  # Create test data:
  python generate_tfrecord.py --csv_input=data/test_labels.csv  --output_path=test.record
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd
import tensorflow as tf
import tensorflow.compat.v1 as tf
from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

flags = tf.app.flags
flags.DEFINE_string('csv_input', './beverage/csv/eval_labels.csv', 'Path to the CSV input')
flags.DEFINE_string('output_path', './beverage/tfrecord/eval.tfrecord', 'Path to output TFRecord')
flags.DEFINE_string('image_dir', './beverage/Augmentations/eval_images', 'Path to images')
FLAGS = flags.FLAGS


def class_text_to_int(row_label):
    if row_label == '2%':
        return 1

    elif row_label == '2%_peach_pet':
        return 2

    elif row_label == 'aloe_pet':
        return 3

    elif row_label == 'apple_pet':
        return 4

    elif row_label == 'coca_can':
        return 5

    elif row_label == 'coca_pet':
        return 6

    elif row_label == 'dailyC':
        return 7

    elif row_label == 'demisoda':
        return 8

    elif row_label == 'fanta_orange_pet':
        return 9

    elif row_label == 'fanta_pine_pet':
        return 10

    elif row_label == 'gal_bae':
        return 11

    elif row_label == 'gal_bae_pet':
        return 12

    elif row_label == 'gas_hwal':
        return 13

    elif row_label == 'grape':
        return 14

    elif row_label == 'hongsam':
        return 15

    elif row_label == 'jeju_pet':
        return 16

    elif row_label == 'lemonade':
        return 17

    elif row_label == 'mango':
        return 18

    elif row_label == 'mccol':
        return 19

    elif row_label == 'milkis':
        return 20

    elif row_label == 'milkis_pet':
        return 21

    elif row_label == 'mogumogu':
        return 22

    elif row_label == 'oranC':
        return 23

    elif row_label == 'peach':
        return 24

    elif row_label == 'pepsi_can':
        return 25

    elif row_label == 'pepsi_pet':
        return 26

    elif row_label == 'pocari':
        return 27

    elif row_label == 'power':
        return 28

    elif row_label == 'red_bull':
        return 29

    elif row_label == 'sol':
        return 30

    elif row_label == 'sprite':
        return 31

    elif row_label == 'sprite_pet':
        return 32

    elif row_label == 'tejava':
        return 33

    elif row_label == 'virak_pet':
        return 34

    elif row_label == 'vita_500':
        return 35

    elif row_label == 'welchs':
        return 36

    else:
        None


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    path = os.path.join(FLAGS.image_dir)
    examples = pd.read_csv(FLAGS.csv_input)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = os.path.join(os.getcwd(), FLAGS.output_path)
    print('Successfully created the TFRecords: {}'.format(output_path))


if __name__ == '__main__':
    tf.app.run()