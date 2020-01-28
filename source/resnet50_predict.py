import tensorflow as tf
import glob
import cv2
import random
import numpy as np
from tensorflow.keras.models import model_from_json

H5_PATH = '/home/barcelona/pervinco/model/four_shapes/2020.01.28_12:22/CAM.h5'
JSON_PATH = '/home/barcelona/pervinco/model/four_shapes/2020.01.28_12:22/CAM.json'
img_path = "/home/barcelona/pervinco/datasets/four_shapes/test/*"
IMG_SIZE = 224


def choice_img(img_path):
    img_list = []
    labels = glob.glob(img_path) # output : /dir/label
    print(len(labels))
    for label in labels:
        imgs = glob.glob(label + '/*.png')
        print('images num :', len(imgs))
        img = random.choice(imgs)
        print(img)
        img_list.append(img)
    return img_list


def load_model(json_path, h5_path):
    with open(json_path, "r") as f:
        loaded_model_json = f.read()

    tl_model = model_from_json(loaded_model_json)
    tl_model.load_weights(h5_path)

    return tl_model


def preprocess_input(img_path):
    # img = pil_image.open(img_path).resize((IMG_SIZE, IMG_SIZE))
    print('input image = ', img_path)
    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    # img_arr = np.asarray(img)[:, :, :3] / 255.
    img_arr = img / 255.
    img_tensor = np.expand_dims(img_arr, 0)

    return img_arr, img_tensor


if __name__ == "__main__":
    # 1. load model
    model = load_model(JSON_PATH, H5_PATH)
    imgs = choice_img(img_path)
    class_list = []

    for l in imgs:
        label = l.split('/')[-2]
        class_list.append(label)
    class_list = sorted(class_list)
    print(class_list)

    for img in imgs:
        img_arr, img_tensor = preprocess_input(img)
        predictions = model.predict(img_tensor)
        print(predictions[0])
        print(class_list[np.argmax(predictions[0])])