import os
import cv2
import math
import numpy as np
import tensorflow as tf
from PIL import Image


def gaussian_radius(det_size, min_overlap=0.7):
  height, width = det_size

  a1  = 1
  b1  = (height + width)
  c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
  sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
  r1  = (b1 + sq1) / 2

  a2  = 4
  b2  = 2 * (height + width)
  c2  = (1 - min_overlap) * width * height
  sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
  r2  = (b2 + sq2) / 2

  a3  = 4 * min_overlap
  b3  = -2 * min_overlap * (height + width)
  c3  = (min_overlap - 1) * width * height
  sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
  r3  = (b3 + sq3) / 2

  return min(r1, r2, r3)


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0

    return h


def draw_gaussian(heatmap, center, radius, k=1):
  diameter = 2 * radius + 1
  gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
  
  x, y = int(center[0]), int(center[1])

  height, width = heatmap.shape[0:2]
    
  left, right = min(x, radius), min(width - x, radius + 1)
  top, bottom = min(y, radius), min(height - y, radius + 1)

  masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
  masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
  
  if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0: # TODO debug
    np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    
  return heatmap


def image_preporcess(image, target_size, gt_boxes=None):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

    ih, iw    = target_size
    h,  w, _  = image.shape

    scale = min(iw/w, ih/h)
    nw, nh  = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0, dtype=np.float32)
    dw, dh = (iw - nw) // 2, (ih-nh) // 2
    image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized
    image_paded = image_paded / 255.

    if gt_boxes is None:
        return image_paded

    else:
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
        return image_paded, gt_boxes


class Datasets(tf.keras.utils.Sequence):
    def __init__(self, txt_path, input_shape, batch_size, num_classes, max_detections=10):
        self.lines = open(txt_path, "r").readlines()
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.max_detections = max_detections
        self.output_shape = (input_shape[0] // 4, input_shape[1] // 4, num_classes)


    def __len__(self):
        return len(self.lines)


    def __getitem__(self, index):
        batch_image = np.zeros((self.batch_size, self.input_shape[0], self.input_shape[1], 3), dtype=np.float32)
        batch_hm = np.zeros((self.batch_size, self.output_shape[0], self.output_shape[1], self.num_classes), dtype=np.float32)
        batch_wh = np.zeros((self.batch_size, self.max_detections, 2), dtype=np.float32)
        batch_reg = np.zeros((self.batch_size, self.max_detections, 2), dtype=np.float32)
        batch_reg_mask = np.zeros((self.batch_size, self.max_detections), dtype=np.float32)
        batch_ind = np.zeros((self.batch_size, self.max_detections), dtype=np.float32)

        for b, i in enumerate(range(index * self.batch_size, (index + 1) * self.batch_size)):
            i = i % len(self.lines)
            try:
                image, hm, wh, reg, reg_mask, ind = self.process_data(i)
                # print(image.shape, hm.shape, wh.shape, reg.shape, reg_mask.shape)
                batch_image[b, :, :, :] = image
                batch_hm[b, :, :, :] = hm
                batch_wh[b, :, :] = wh
                batch_reg[b, :, :] = reg
                batch_reg_mask[b, :] = reg_mask
                batch_ind[b, :] = ind

            except:
                pass

        return (batch_image, batch_hm, batch_wh, batch_reg, batch_reg_mask, batch_ind), np.zeros((self.batch_size))


    def process_data(self, idx):
        line = self.lines[idx]
        s = line.split()
        
        image_path = s[0]
        image = np.array(cv2.imread(image_path))
        # labels = np.array([list(map(lambda x: int(float(x)), box.split(','))) for box in s[1:]])
        labels = np.array([np.array(list(map(int,box.split(',')))) for box in s[1:]])
        
        if len(labels) > 0:
            image, labels = image_preporcess(np.copy(image), [self.input_shape[0], self.input_shape[1]], np.copy(labels))
            
            output_h = self.input_shape[0] // 4
            output_w = self.input_shape[1] // 4
            hm = np.zeros((output_h, output_w, self.num_classes),dtype=np.float32)
            wh = np.zeros((self.max_detections, 2),dtype=np.float32)
            reg = np.zeros((self.max_detections, 2),dtype=np.float32)
            ind = np.zeros((self.max_detections),dtype=np.float32)
            reg_mask = np.zeros((self.max_detections),dtype=np.float32)

            for idx, label in enumerate(labels):
                bbox = label[:4]/4
                class_id = label[4]
                h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
                radius = gaussian_radius((math.ceil(h),math.ceil(w)))
                radius = max(0, int(radius))
                ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                draw_gaussian(hm[:,:,class_id], ct_int, radius)
                wh[idx] = 1. * w, 1. * h
                ind[idx] = ct_int[1] * output_w + ct_int[0]
                reg[idx] = ct - ct_int
                reg_mask[idx] = 1
            
            return image, hm, wh, reg, reg_mask, ind