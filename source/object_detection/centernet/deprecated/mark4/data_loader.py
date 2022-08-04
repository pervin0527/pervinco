import os
import cv2
import math
import numpy as np
import tensorflow as tf
from data_utils import image_preporcess, gaussian_radius, draw_umich_gaussian

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, txt_file, classes, batch_size, input_shape, max_detections):
        self.total_lines = open(txt_file, "r").readlines()
        self.classes = classes
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.max_detections = max_detections

    def __getitem__(self, index):
        batch_image = np.zeros((self.batch_size, self.input_shape[0], self.input_shape[1], 3), dtype=np.float32)
        batch_hm = np.zeros((self.batch_size, self.input_shape[0] // 4, self.input_shape[1] // 4, len(self.classes)), dtype=np.float32)
        batch_wh = np.zeros((self.batch_size, self.max_detections, 2), dtype=np.float32)
        batch_reg = np.zeros((self.batch_size, self.max_detections, 2), dtype=np.float32)
        batch_reg_mask = np.zeros((self.batch_size, self.max_detections), dtype=np.float32)
        batch_ind = np.zeros((self.batch_size, self.max_detections), dtype=np.float32)

        for num, i in enumerate(range(index * self.batch_size, (index + 1) * self.batch_size)):
            i = i % len(self.total_lines)
            image, hm, wh, reg, reg_mask, ind = self.process_data(self.total_lines[i], self.input_shape)
            batch_image[num, :, :, :] = image
            batch_hm[num, :, :, :] = hm
            batch_wh[num, :, :] = wh
            batch_reg[num, :, :] = reg
            batch_reg_mask[num, :] = reg_mask
            batch_ind[num, :] = ind

        # return batch_image, batch_hm, batch_wh, batch_reg, batch_reg_mask, batch_ind
        # return [batch_image, batch_hm, batch_wh, batch_reg, batch_reg_mask, batch_ind], np.zeros((self.batch_size,))
        return batch_image, [batch_hm, batch_wh, batch_reg, batch_reg_mask, batch_ind]

    def process_data(self, line, input_shape):
        line = line.strip().split(' ')

        image = np.array(cv2.imread(line[0]))
        labels = np.array([list(map(lambda x: int(float(x)), box.split(','))) for box in line[1:]])
        image, labels = image_preporcess(np.copy(image), [input_shape[0], input_shape[1]], np.copy(labels))

        output_h, output_w = input_shape[0] // 4, input_shape[1] // 4
        hm = np.zeros((output_h, output_w, len(self.classes)), dtype=np.float32)
        wh = np.zeros((self.max_detections, 2), dtype=np.float32)
        reg = np.zeros((self.max_detections, 2), dtype=np.float32)
        ind = np.zeros((self.max_detections), dtype=np.float32)
        reg_mask = np.zeros((self.max_detections), dtype=np.float32)

        for idx, label in enumerate(labels):
            bbox = label[:4] / 4
            class_id = label[4]
            
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            radius = gaussian_radius((math.ceil(h), math.ceil(w)))
            radius = max(0, int(radius))

            ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
            ct_int = ct.astype(np.int32)

            draw_umich_gaussian(hm[:, :, class_id], ct_int, radius)
            wh[idx] = 1. * w, 1. * h
            ind[idx] = ct_int[1] * output_w + ct_int[0]
            reg[idx] = ct - ct_int
            reg_mask[idx] = 1

        return image, hm, wh, reg, reg_mask, ind

    def __len__(self):
        return len(self.total_lines)