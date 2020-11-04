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


# GPU control
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=7000)])
  except RuntimeError as e:
      print(e)


def set_mode(load_map_file):
    df = pd.read_csv(load_map_file, sep = ' ', index_col=False, header=None)
    file_list = df[0].tolist()
    # print(file_list)

    main_model_path = file_list[0]
    empty_model_path = file_list[1]
    product_xml = file_list[2]
    empty_xml = file_list[3]
    main_label_file = file_list[4]
    binary_label_file = file_list[5]
    mode = file_list[6]

    return main_model_path, empty_model_path, product_xml, empty_xml, main_label_file, binary_label_file, mode


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


def inference(image, results, start_time):
    imgs = crop_image(image, main_boxes, (224, 224))
    em_imgs = crop_image(image, em_boxes, (224, 224))
    
    for img, em_img, idx in zip(imgs, em_imgs, range(len(imgs))):
        main_result = CLASS_NAMES[predict(img, main_model, True)]
        empty_result = EM_CLASS_NAMES[predict(em_img, empty_model, False)]
        results[idx] = empty_result if empty_result == 'empty' else main_result

    end_time = time.time() - start_time
    return results, end_time


def test_inference(image, results, start_time):
    imgs = crop_image(image, main_boxes, (224, 224))
    em_imgs = crop_image(image, em_boxes, (224, 224))
    
    for img, em_img, idx in zip(imgs, em_imgs, range(len(imgs))):
        empty_result = EM_CLASS_NAMES[predict(em_img, empty_model, False)]

        if empty_result == "empty":
            results[idx] = empty_result

        else:
            main_result = CLASS_NAMES[predict(img, main_model, True)]
            results[idx] = main_result


    end_time = time.time() - start_time
    return results, end_time


def main_inference(image, results, start_time):
    imgs = crop_image(image, main_boxes, (224, 224))

    for img, idx in zip(imgs, range(len(imgs))):
        main_result = CLASS_NAMES[predict(img, main_model, True)]
        results[idx] = main_result

    print(time.time() - start_time)
    return results

def empty_inference(image, results, start_time):
    em_imgs = crop_image(image, em_boxes, (224, 224))

    for em_img, idx in zip(em_imgs, range(len(em_imgs))):
        empty_result = EM_CLASS_NAMES[predict(em_img, empty_model, False)]
        results[idx] = empty_result

    print(time.time() - start_time)
    return results
        

def predict(img, model, test):
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    predictions = model.predict(img, steps=1)
    score = np.argmax(predictions[0])
    return score


def init_model(test_img):
    img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)
    
    return img


def preprocess_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = tf.keras.applications.efficientnet.preprocess_input(image)

    return image


if __name__ == "__main__":
    load_map_path = sys.argv[1]
    main_model_path, empty_model_path, main_xml, empty_xml, main_label_file, binary_label_file, m = set_mode(load_map_path)

    print(main_model_path, empty_model_path, main_xml, empty_xml, main_label_file, binary_label_file, m)

    empty_model = tf.keras.models.load_model(empty_model_path)
    main_model = tf.keras.models.load_model(main_model_path)

    df = pd.read_csv(main_label_file, sep = ' ', index_col=False, header=None)
    CLASS_NAMES = df[0].tolist()
    CLASS_NAMES = sorted(CLASS_NAMES)

    df = pd.read_csv(binary_label_file, sep = ' ', index_col=False, header=None)
    EM_CLASS_NAMES = df[0].tolist()
    EM_CLASS_NAMES = sorted(EM_CLASS_NAMES)

    em_boxes = get_boxes(empty_xml)
    main_boxes = get_boxes(main_xml)
    print(len(em_boxes))
    print(len(main_boxes))

    results = ['' for i in range(len(main_boxes))]

    test = init_model(np.zeros((224, 224, 3), np.uint8))
    empty_model.predict(test, steps=1)
    main_model.predict(test, steps=1)

    test_images = sorted(glob.glob('/data/backup/pervinco_2020/test_code/test_image/emart24/*.jpg'))
    os.system('clear')

    # for frame in test_images:
    #     file_name = frame.split('/')[-1]
    #     start_time = time.time()
    #     frame = cv2.imread(frame)
    #     main_results = main_inference(frame, results, start_time)
    #     print(main_results)

    # for frame in test_images:
    #     file_name = frame.split('/')[-1]
    #     start_time = time.time()
    #     frame = cv2.imread(frame)
    #     empty_results = empty_inference(frame, results, start_time)
    #     print(empty_results)

    # total_time = 0
    # for frame in test_images:
    #     file_name = frame.split('/')[-1]
    #     start_time = time.time()
    #     frame = cv2.imread(frame)
    #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     results, end_time = inference(frame, results, start_time)
    #     print(results)

    #     total_time += end_time

    # print(total_time)

    # total_time = 0
    # for frame in test_images:
    #     file_name = frame.split('/')[-1]
    #     start_time = time.time()
    #     frame = cv2.imread(frame)
    #     results, end_time = inference(frame, results, start_time)
    #     print(results)

    #     total_time += end_time

    # print(total_time)

    total_time = 0
    for frame in test_images:
        file_name = frame.split('/')[-1]
        start_time = time.time()
        frame = cv2.imread(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results, end_time = test_inference(frame, results, start_time)
        print(file_name, results)

        total_time += end_time

    print(total_time)