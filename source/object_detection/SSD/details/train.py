import cv2
import os
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw

def transform_center_to_corner(boxes):
    corner_box = tf.concat([boxes[..., :2] - boxes[..., 2:] / 2, boxes[..., :2] + boxes[..., 2:] / 2], axis=-1)

    return corner_box

def compute_target(default_boxes, gt_boxes, gt_labels, iou_threshold=0.5):
    transformed_default_boxes = transform_center_to_corner(default_boxes)

    iou = compute_iou(transformed_default_boxes, gt_boxes)

    best_gt_iou = tf.math.reduce_max(iou, 1)
    best_gt_idx = tf.math.argmax(iou, 1)

    best_default_iou = tf.math.reduce_max(iou, 0)
    best_default_idx = tf.math.argmax(iou, 0)

    best_gt_idx = tf.tensor_scatter_nd_update(best_gt_idx,
                                              tf.expand_dims(best_default_idx, 1),
                                              tf.range(best_default_idx.shape[0], dtype=tf.int64))

    best_gt_iou = tf.tensor_scatter_nd_update(best_gt_iou,
                                              tf.expand_dims(best_default_idx, 1),
                                              tf.ones_like(best_default_idx, dtype=tf.float32))

    gt_confs = tf.gather(gt_labels, best_gt_idx)
    gt_confs = tf.where(tf.less(best_gt_iou, iou_threshold), tf.zeros_like(gt_confs), gt_confs)

    gt_boxes = tf.gather(gt_boxes, best_gt_idx)
    gt_locs = encode(default_boxes, gt_boxes)

    return gt_confs, gt_locs

def get_image(index):
    filename = ids[index]
    img_path = os.path.join(image_dir, filename+'.jpg')
    img = Image.open(img_path)

    return img

def get_annotation(index, orig_shape):
    h, w = orig_shape
    filename = ids[index]

    anno_path = os.path.join(anno_dir, filename+'.xml')
    objects = ET.parse(anno_path).findall('object')

    boxes = []
    labels = []

    for obj in objects:
        name = obj.find('name').text.lower().strip()
        bndbox = obj.find('bndbox')
        xmin = (float(bndbox.find('xmin').text) - 1) / w
        ymin = (float(bndbox.find('ymin').text) - 1) / h
        xmax = (float(bndbox.find('xmax').text) - 1) / w
        ymax = (float(bndbox.find('ymax').text) - 1) / h
        boxes.append([xmin, ymin, xmax, ymax])

        labels.append(name_to_idx[name] + 1)

    return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64)

if __name__ == "__main__":
    data_dir = "/data/Datasets/Seeds/traffic_sign"
    
    label_file = pd.read_csv(f'{data_dir}/labels.txt', sep=' ', index_col=False, header=None)
    idx_to_name = label_file[0].tolist()
    name_to_idx = dict([v, k] for k, v in enumerate(idx_to_name))
    print(idx_to_name)
    print(name_to_idx)

    image_dir = os.path.join(data_dir, 'images')
    anno_dir = os.path.join(data_dir, 'annotations')
    
    ids = list(map(lambda x: x[:-4], os.listdir(image_dir)))
    train_ids = ids[:int(len(ids) * 0.75)]
    val_ids = ids[int(len(ids) * 0.75):]

    for index in range(len(ids)):
        filename = ids[index]
        img = get_image(index)
        w, h = img.size

        boxes, labels = get_annotation(index, (h, w))
        print(boxes, labels)

        for coords in boxes:
            draw = ImageDraw.Draw(img)
            draw.rectangle(coords, outline='red', width=3)

        img = np.array(img.resize((300, 300)), dtype=np.float32)
        img = (img / 127.0) - 1.0

        # img.show()
        cv2.imshow('image', img)
        cv2.waitKey(0)
        break