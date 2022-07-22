import os
import cv2
import math
import numpy as np
import tensorflow as tf
from PIL import Image

def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 


def preprocess_input(image):
    image = np.array(image, dtype = np.float32)[:, :, ::-1]
    mean = [0.40789655, 0.44719303, 0.47026116]
    std = [0.2886383, 0.27408165, 0.27809834]

    return (image / 255. - mean) / std


def draw_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    
    return heatmap


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0

    return h


def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2

    return min(r1, r2, r3)


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
        batch_images = np.zeros((self.batch_size, self.input_shape[0], self.input_shape[1], 3), dtype=np.float32)
        batch_hms = np.zeros((self.batch_size, self.output_shape[0], self.output_shape[1], self.num_classes), dtype=np.float32)
        batch_whs = np.zeros((self.batch_size, self.max_detections, 2), dtype=np.float32)
        batch_regs = np.zeros((self.batch_size, self.max_detections, 2), dtype=np.float32)
        batch_reg_masks = np.zeros((self.batch_size, self.max_detections), dtype=np.float32)
        batch_indices = np.zeros((self.batch_size, self.max_detections), dtype=np.float32)

        for b, i in enumerate(range(index * self.batch_size, (index + 1) * self.batch_size)):
            i = i % len(self.lines)
            image, box = self.get_random_data(self.lines[i], self.input_shape)

            if len(box) != 0:
                boxes = np.array(box[:, :4],dtype=np.float32)
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

                # batch_images[b] = preprocess_input(np.array(image, np.float32))
                batch_images[b] = np.array(image, dtype=np.float32)
                batch_images = (batch_images / 127.5) - 1

            return [batch_images, batch_hms, batch_whs, batch_regs, batch_reg_masks, batch_indices], np.zeros((self.batch_size,))


    def get_random_data(self, lines, input_shape):
        line = lines.split()

        image = Image.open(line[0])
        image = cvtColor(image)
        img_height, img_width = image.size
        h, w = input_shape[:2]

        box = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])

        scale = min(w / img_width, h / img_height)
        nw = int(img_width * scale)
        nh = int(img_height * scale)
        dx = (w - nw) // 2
        dy = (h - nh) // 2

        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new("RGB", (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image_data = np.array(new_image, np.float32)

        if len(box)>0:
            np.random.shuffle(box)
            box[:, [0,2]] = box[:, [0,2]]*nw/img_width + dx
            box[:, [1,3]] = box[:, [1,3]]*nh/img_height + dy
            box[:, 0:2][box[:, 0:2]<0] = 0
            box[:, 2][box[:, 2]>w] = w
            box[:, 3][box[:, 3]>h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box

        return image_data, box


if __name__ == "__main__":
    DATA_DIR = "/home/ubuntu/Datasets/WIDER/CUSTOM"
    train_txt = f"{DATA_DIR}/train/list.txt"
    test_txt = f"{DATA_DIR}/test/list.txt"

    train_dataset = Datasets(train_txt, (512, 512, 3), 1, 1, True, 10)
    data = train_dataset[0]

    for i in range(len(data)):
        print(data[i].shape)