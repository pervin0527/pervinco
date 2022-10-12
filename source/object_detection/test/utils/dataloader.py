import cv2
import math
import numpy as np
import tensorflow as tf
from PIL import Image
from utils.utils import cvtColor, preprocess_input

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, annotation_lines, input_shape, batch_size, num_classes, max_detections):
        self.annotation_lines = annotation_lines
        self.length = len(self.annotation_lines)
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.max_detections = max_detections

    def __len__(self):
        return math.ceil(len(self.annotation_lines) / float(self.batch_size))

    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a

    def __getitem__(self, index):
        image_data, box_data = [], []

        for i in range(index * self.batch_size, (index + 1) * self.batch_size):
            i = i % self.length
            image, box = self.get_random_data(self.annotation_lines[i], self.input_shape, self.max_detections)

            if len(box) != 0:
                box[:, 2:4] = box[:, 2:4] - box[:, 0:2]
                box[:, 0:2] = box[:, 0:2] + box[:, 2:4] / 2

            image_data.append(preprocess_input(np.array(image, np.float32)))
            box_data.append(box)
        
        image_data = np.array(image_data)
        box_data = np.array(box_data)

        return [image_data, box_data], np.zeros(self.batch_size)

    def get_random_data(self, annotation_line, input_shape, max_detections):
        line = annotation_line.split()

        image = Image.open(line[0])
        image = cvtColor(image)

        iw, ih = image.size
        h, w = input_shape

        box = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)
        dx = (w-nw)//2
        dy = (h-nh)//2

        image = image.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_image.paste(image, (dx, dy))
        image_data = np.array(new_image, np.float32)

        box_data = np.zeros((max_detections,5))
        if len(box)>0:
            np.random.shuffle(box)
            box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
            box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
            box[:, 0:2][box[:, 0:2]<0] = 0
            box[:, 2][box[:, 2]>w] = w
            box[:, 3][box[:, 3]>h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w>1, box_h>1)]
            if len(box)>max_detections: box = box[:max_detections]
            box_data[:len(box)] = box

        return image_data, box_data