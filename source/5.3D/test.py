import os, sys, h5py
import numpy as np
import tensorflow as tf

BATCH_SIZE = 32
NUM_POINT = 1024
DECAY_STEP = 200000
BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

TRAIN_FILES = "/home/barcelona/pervinco/source/5.3D/pointnet/tf2/data/modelnet40_ply_hdf5_2048/train_files.txt"
VALID_FILES = "/home/barcelona/pervinco/source/5.3D/pointnet/tf2/data/modelnet40_ply_hdf5_2048/test_files.txt"

TRAIN_FILES = [line.rstrip() for line in open(TRAIN_FILES)]
VALID_FILES = [line.rstrip() for line in open(VALID_FILES)]

print(TRAIN_FILES)
print(VALID_FILES)

def get_model(point_cloud, is_training, bn_decay=None):
    batch_size = point_cloud.get_shape()[0]
    num_point = point_cloud.get_shape()[1]

    print(batch_size, num_point)

    end_points = {}

    return 0, 0

with tf.Graph().as_default():
    pointclouds_pl = tf.Variable(initial_value=tf.zeros([BATCH_SIZE, NUM_POINT, 3]))
    labels_pl = tf.Variable(initial_value=tf.zeros([BATCH_SIZE]))

    print(pointclouds_pl, labels_pl)

    pred, end_points = get_model(pointclouds_pl, True)