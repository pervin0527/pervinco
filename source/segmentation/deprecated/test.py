import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

mask = "/data/Datasets/VOCdevkit/VOC2012/SegmentationRaw/2007_000032.png"

####################################################################################
LABEL_PATH = "/data/Datasets/VOCdevkit/VOC2012/Labels/class_labels.txt"
label_df = pd.read_csv(LABEL_PATH, lineterminator='\n', header=None, index_col=False)
CLASSES = label_df[0].to_list()
NUM_CLASSES = len(CLASSES)
print(CLASSES)
CLASS_VALUES = [CLASSES.index(cls.lower()) for cls in CLASSES]

COLORMAP = [[0, 0, 0], # background
            [128, 0, 0], # aeroplane
            [0, 128, 0], # bicycle
            [128, 128, 0], # bird
            [0, 0, 128], # boat
            [128, 0, 128], # bottle
            [0, 128, 128], # bus
            [128, 128, 128], # car
            [64, 0, 0], # cat
            [192, 0, 0], # chair
            [64, 128, 0], # cow
            [192, 128, 0], # diningtable
            [64, 0, 128], # dog
            [192, 0, 128], # horse
            [64, 128, 128], # motorbike
            [192, 128, 128], # person
            [0, 64, 0], # potted plant
            [128, 64, 0], # sheep
            [0, 192, 0], # sofa
            [128, 192, 0], # train
            [0, 64, 128] # tv/monitor
]
####################################################################################

# mask = cv2.imread(mask, 0)
# print(mask.shape)

# one_hot = np.zeros((mask.shape[0], mask.shape[1], n_classes))
# for i, unique_value in enumerate(np.unique(mask)):
#     one_hot[:, :, i][mask == unique_value] = 1

# print(one_hot.shape)

mask = tf.io.read_file(mask)
mask = tf.image.decode_png(mask, 1)

mask = tf.squeeze(mask, axis=-1)
# mask = tf.ones((443, 300), dtype=tf.int32)
print(mask.shape)

one_hot_mask = tf.one_hot(mask, NUM_CLASSES)
print(one_hot_mask.shape)