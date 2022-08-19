import os
import sys
import cv2
import yaml
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from pathlib import Path
from magic_point_model import MagicPoint
from data_utils import homography_adaptation, box_nms


def draw_keypoints(img, corners, color):
    keypoints = [cv2.KeyPoint(int(c[1]), int(c[0]), 1) for c in np.stack(corners).T]
    return cv2.drawKeypoints(img.astype(np.uint8), keypoints, None, color=color)


if __name__ == "__main__":
    config_path = "./coco_export_test.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model = MagicPoint("vgg", (240, 320), 4, 0.001)
    model.built = True
    model.load_weights("/home/ubuntu/Models/MagicPoint/vgg.h5")

    image_path = "/home/ubuntu/Datasets/COCO2014/train2014/COCO_train2014_000000519723.jpg"
    image = tf.io.read_file(image_path)
    image = tf.io.decode_jpeg(image, channels=1)
    image = tf.image.resize(image, (240, 320))
    input_image = tf.cast(image, tf.float32) / 255.

    outputs = homography_adaptation(input_image, model, config)
    outputs ={k: v[0] for k, v in outputs.items()}  # batch to single element
    outputs['prob_nms'] = box_nms(outputs['prob'], config["model"]['nms_size'], keep_top_k=config["model"]['top_k'])
    outputs['pred'] = tf.cast(tf.greater_equal(outputs['prob_nms'], config["model"]['threshold']), dtype=np.int32)

    image = image.numpy()
    pred = outputs["pred"].numpy()
    result = draw_keypoints(image, np.where(pred), (0, 255, 0))

    cv2.imwrite("./samples/image.jpg", image)
    cv2.imwrite("./samples/result.jpg", result)