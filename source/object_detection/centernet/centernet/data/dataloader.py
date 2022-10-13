import math
import numpy as np
import tensorflow as tf
from PIL import Image
from data.data_utils import cvtColor, gaussian_radius, draw_gaussian, preprocess_input

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, lines, input_shape, batch_size, num_classes, is_train, max_detections):
        self.lines = lines
        self.length = len(self.lines)
        self.input_shape = input_shape
        self.output_shape = (int(input_shape[0] / 4), int(input_shape[1] / 4))
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.is_train = is_train
        self.max_detections = max_detections

    def get_random_data(self, annotation_line, input_shape):
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

        image       = image.resize((nw,nh), Image.BICUBIC)
        new_image   = Image.new('RGB', (w,h), (128,128,128))
        new_image.paste(image, (dx, dy))
        image_data  = np.array(new_image, np.float32)

        if len(box)>0:
            np.random.shuffle(box)
            box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
            box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
            box[:, 0:2][box[:, 0:2]<0] = 0
            box[:, 2][box[:, 2]>w] = w
            box[:, 3][box[:, 3]>h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box

        return image_data, box

    def __getitem__(self, index):
        batch_images = np.zeros((self.batch_size, self.input_shape[0], self.input_shape[1], 3), dtype=np.float32)
        batch_hms = np.zeros((self.batch_size, self.output_shape[0], self.output_shape[1], self.num_classes), dtype=np.float32)
        batch_whs = np.zeros((self.batch_size, self.max_detections, 2), dtype=np.float32)
        batch_regs = np.zeros((self.batch_size, self.max_detections, 2), dtype=np.float32)
        batch_reg_masks = np.zeros((self.batch_size, self.max_detections), dtype=np.float32)
        batch_indices = np.zeros((self.batch_size, self.max_detections), dtype=np.float32)

        for b, i in enumerate(range(index * self.batch_size, (index + 1) * self.batch_size)):
            i = i % self.length
            image, box = self.get_random_data(self.lines[i], self.input_shape)

            if len(box) != 0:
                boxes = np.array(box[:, :4], dtype=np.float32)
                boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]] / self.input_shape[1] * self.output_shape[1], 0, self.output_shape[1] - 1)
                boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]] / self.input_shape[0] * self.output_shape[0], 0, self.output_shape[0] - 1)

            for i in range(len(box)):
                bbox = boxes[i].copy()
                cls_id = int(box[i, -1])
                
                h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
                if h > 0 and w > 0:
                    radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                    radius = max(0, int(radius))

                    ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                    ct_int = ct.astype(np.int32)

                    batch_hms[b, :, :, cls_id] = draw_gaussian(batch_hms[b, :, :, cls_id], ct_int, radius)
                    batch_whs[b, i] = 1. * w, 1. * h
                    batch_regs[b, i] = ct - ct_int
                    batch_reg_masks[b, i] = 1
                    batch_indices[b, i] = ct_int[1] * self.output_shape[0] + ct_int[0]

            # batch_images[b] = preprocess_input(image)
            batch_images[b] = image
            # batch_images[b] = image / 127.5 -1

        return [batch_images, batch_hms, batch_whs, batch_regs, batch_reg_masks, batch_indices], np.zeros((self.batch_size,))

    def __len__(self):
        return math.ceil(len(self.lines) / self.batch_size)