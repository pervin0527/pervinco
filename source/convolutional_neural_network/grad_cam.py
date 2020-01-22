# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image as pil_image
import cv2
import glob
import tensorflow as tf
from tensorflow.python.keras.models import model_from_json
from tensorflow.python.keras import backend as K
from tensorflow.python.framework import ops
import scipy
from scipy import ndimage
from skimage.measure import label, regionprops

H5_PATH = "/home/barcelona/pervinco/model/cats_and_dogs/2020.01.22_11:26/CAM.h5"
JSON_PATH = "/home/barcelona/pervinco/model/cats_and_dogs/2020.01.22_11:26/CAM.json"
IMG_SIZE = 299


def load_model(json_path, h5_path):
    with open(json_path, "r") as f:
        loaded_model_json = f.read()

    tl_model = model_from_json(loaded_model_json)
    tl_model.load_weights(h5_path)

    return tl_model


def preprocess_input(img_path):
    img = pil_image.open(img_path).resize((IMG_SIZE, IMG_SIZE))
    img_arr = np.asarray(img)[:, :, :3] / 255.
    img_tensor = np.expand_dims(img_arr, 0)

    return img_arr, img_tensor


def generate_cam(model, img_path, class_idx):
    # img_path -> preprocessed image tensor
    img_arr, img_tensor = preprocess_input(img_path)
    # model.summary()

    # preprocessed image tensor -> last_conv_output, predictions
    get_output = K.function([model.layers[0].input], [model.layers[-4].output, model.layers[-1].output])
    [conv_outputs, predictions] = get_output([img_tensor])

    conv_outputs = conv_outputs[0, :, :, :]
    class_weights = model.layers[-1].get_weights()[0]

    # generate cam
    cam = np.zeros(dtype=np.float32, shape=conv_outputs.shape[0:2])
    for i, w in enumerate(class_weights[:, class_idx]):
        cam += w * conv_outputs[:, :, i]

    cam /= np.max(cam)
    cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))

    return img_arr, cam, predictions


def generate_bbox(img, cam, threshold):
    labeled, nr_objects = ndimage.label(cam > threshold)
    props = regionprops(labeled)
    return props


if __name__ == "__main__":
    # 1. load model
    model = load_model(JSON_PATH, H5_PATH)

    # 2. image sources
    samples = []
    class_indices = ['cat', 'dog']

    for idx, ci in enumerate(class_indices):
        print(ci, idx)
        tmp_dict = {}
        tmp_dict['target'] = ci
        tmp_dict['img_path'] = '/home/barcelona/pervinco/datasets/cats_and_dogs_small_set/test/' + ci + '.jpg'
        tmp_dict['class_idx'] = idx
        samples.append(tmp_dict)

    fig, axes = plt.subplots(4, 2, figsize=(20, 20))

    for i, s in enumerate(samples):
        img_set = s['target']
        img_path = s['img_path']
        class_idx = s['class_idx']
        img, cam, predictions = generate_cam(model, img_path, class_idx)
        pred_values = np.squeeze(predictions, 0)
        top1 = np.argmax(pred_values)
        top1_value = round(float(pred_values[top1]), 3)
        props = generate_bbox(img, cam, 0.7)

        axes[0, i].imshow(img)
        axes[1, i].imshow(cam)
        axes[2, i].imshow(img)
        axes[2, i].imshow(cam, cmap='jet', alpha=0.5)

        axes[3, i].imshow(img)
        for b in props:
            bbox = b.bbox
            xs = bbox[1]
            ys = bbox[0]
            w = bbox[3] - bbox[1]
            h = bbox[2] - bbox[0]

            rect = patches.Rectangle((xs, ys), w, h, linewidth=2, edgecolor='r', facecolor='none')
            axes[3, i].add_patch(rect)

        # axes[0, i].axis('off')
        # axes[1, i].axis('off')
        # axes[2, i].axis('off')
        # axes[3, i].axis('off')

        # axes[0, i].set_title("pred: {} - {}".format(class_indices[top1], top1_value), fontsize=15)

    plt.tight_layout()
    plt.show()
