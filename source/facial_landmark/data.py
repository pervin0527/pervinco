import cv2
import math
import random
import numpy as np
import tensorflow as tf

class PFLDDatasets(tf.keras.utils.Sequence):
    def __init__(self, data_flie, batch_size):
        self.batch_size = batch_size
        with open(data_flie, 'r', encoding='utf-8') as f:
            self.lines = f.readlines()
            random.shuffle(self.lines)
        self.length = len(self.lines)
        
    def __getitem__(self, index):
        images_list = []
        label_list = []
        for i in range(index * self.batch_size, (index + 1) * self.batch_size):
            i = i % self.length
            line = self.lines[i].strip().split()
            img = cv2.imread(line[0])
            img = img / 255.0
            # img = (img / 127.5) - 1
            label = np.array(line[1:206], dtype=np.float32)
            images_list.append(np.array(img, dtype=np.float32))
            label_list.append(label)

        images_list = np.array(images_list)
        label_list = np.array(label_list)

        return images_list, label_list

    def __len__(self):
        return math.ceil(self.length / float(self.batch_size))