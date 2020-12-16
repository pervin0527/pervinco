import tensorflow as tf
import cv2
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from xml.dom import minidom
from datetime import datetime
import glob
import os
import sys
import time
from tensorflow.keras.models import load_model

os.environ["CUDA_VISIBLE_DEVICES"]="0"
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   try:
#     for gpu in gpus:
#         tf.config.experimental.set_virtual_device_configuration(gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10000)])
#   except RuntimeError as e:
#     print(e)


def crop(image, box):
    xmin, ymin, xmax, ymax = box
    result = image[ymin:ymax+1, xmin:xmax+1, :]
    return result


def crop_image(image, boxes, resize=None, save_path=None):
    images = list(map(lambda b : crop(image, b), boxes)) 

    if str(type(resize)) == "<class 'tuple'>":
        try:
            images = list(map(lambda i: cv2.resize(i, dsize=resize, interpolation=cv2.INTER_LINEAR), images))
        except Exception as e:
            print(str(e))
    return images


def get_boxes(label_path):
    xml_path = os.path.join(label_path)

    root_1 = minidom.parse(xml_path)
    bnd_1 = root_1.getElementsByTagName('bndbox')

    result = []
    for i in range(len(bnd_1)):
        xmin = int(bnd_1[i].childNodes[1].childNodes[0].nodeValue)
        ymin = int(bnd_1[i].childNodes[3].childNodes[0].nodeValue)
        xmax = int(bnd_1[i].childNodes[5].childNodes[0].nodeValue)
        ymax = int(bnd_1[i].childNodes[7].childNodes[0].nodeValue)
        result.append((xmin,ymin,xmax,ymax))
    
    return result


def predict(img, model):
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    predictions = model.predict(img, steps=1)
    score = np.argmax(predictions[0])
    return score


def inference(start_time, frame, main_boxes, empty_boxes):
    main_crops = crop_image(frame, main_boxes, (224, 224))
    empty_crops = crop_image(frame, empty_boxes, (224, 224))

    results = []

    for main_crop, empty_crop, idx in zip(main_crops, empty_crops, range(len(main_crops))):
        empty_result = EM_CLASS_NAMES[predict(empty_crop, empty_model)]

        if empty_result == "empty":
            results.append(empty_result)

        else:
            main_result = CLASS_NAMES[predict(main_crop, main_model)]
            results.append(main_result)

    end_time = time.time() - start_time
    return results, end_time


def init_model(test_img, main_model, empty_model):
    img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)

    empty_model.predict(img, steps=1)
    main_model.predict(img, steps=1)


if __name__ == "__main__":
    empty_model_path = "/data/tf_workspace/source/models/empty_model.h5"
    main_model_path = "/data/tf_workspace/source/models/main_model.h5"
    
    right_main_xml = "/data/tf_workspace/source/bbox/0/r/main.xml"
    right_empty_xml = "/data/tf_workspace/source/bbox/0/r/empty.xml"
    left_main_xml = "/data/tf_workspace/source/bbox/0/l/main.xml"
    left_empty_xml = "/data/tf_workspace/source/bbox/0/l/empty.xml"

    main_label_file = "/data/tf_workspace/source/models/main_labels.txt"
    binary_label_file = "/data/tf_workspace/source/models/empty_labels.txt"

    empty_model = tf.keras.models.load_model(empty_model_path)
    main_model = tf.keras.models.load_model(main_model_path)

    os.system("clear")

    df = pd.read_csv(main_label_file, sep = ' ', index_col=False, header=None)
    CLASS_NAMES = df[0].tolist()
    CLASS_NAMES = sorted(CLASS_NAMES)

    df = pd.read_csv(binary_label_file, sep = ' ', index_col=False, header=None)
    EM_CLASS_NAMES = df[0].tolist()
    EM_CLASS_NAMES = sorted(EM_CLASS_NAMES)

    rm_boxes = get_boxes(right_main_xml)
    rem_boxes = get_boxes(right_empty_xml)
    lm_boxes = get_boxes(left_main_xml)
    lem_boxes = get_boxes(left_empty_xml)

    print("Right boxes", len(rm_boxes), len(rem_boxes))
    print("left boxes", len(lm_boxes), len(lem_boxes))


    left_test_images = sorted(glob.glob('/data/tf_workspace/source/test_images/left/*.jpg'))
    right_test_images = sorted(glob.glob('/data/tf_workspace/source/test_images/right/*.jpg'))

    init_model(np.zeros((224, 224, 3), np.uint8), main_model, empty_model)
    os.system('clear')

    total_time = 0
    for left_frame, right_frame in zip(left_test_images, right_test_images):
        start_time = time.time()
        
        left_frame = cv2.imread(left_frame)
        left_frame = cv2.cvtColor(left_frame, cv2.COLOR_BGR2RGB)

        right_frame = cv2.imread(right_frame)
        right_frame = cv2.cvtColor(right_frame, cv2.COLOR_BGR2RGB)

        left_result, left_end_time = inference(start_time, left_frame, lm_boxes, lem_boxes)
        right_result, right_end_time = inference(start_time, right_frame, rm_boxes, rem_boxes)
        step_time = left_end_time + right_end_time

        print(step_time)
        print(left_result + right_result)
        total_time += step_time

    print(total_time)