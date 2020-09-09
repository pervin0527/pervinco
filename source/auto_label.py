import random
import cv2
import os
import time
import sys
import pandas as pd
import glob
import datetime
import numpy as np
import tensorflow as tf
from efficientnet.tfkeras import preprocess_input
from matplotlib import pyplot as plt
import xml.etree.ElementTree as ET
from xml.dom import minidom


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=9000)])
  except RuntimeError as e:
    print(e)


def crop(image, box):
    xmin, ymin, xmax, ymax = box
    result = image[ymin:ymax+1, xmin:xmax+1, :]
    return result


def crop_image(image, boxes, resize=None, save_path=None):
    # image: cv2 image
    images = list(map(lambda b : crop(image, b), boxes)) 
    # boxes: [[xmin, ymin, xmax, ymax], ...] <- 이걸로 crop

    if str(type(resize)) == "<class 'tuple'>":
        try:
            images = list(map(lambda i: cv2.resize(i, dsize=resize, interpolation=cv2.INTER_LINEAR), images))
        except Exception as e:
            print(str(e))
    return images


def get_boxes(label_path):
    # print(label_path)
    xml_path = os.path.join(label_path)

    root_1 = minidom.parse(xml_path)  # xml.dom.minidom.parse(xml_path)
    bnd_1 = root_1.getElementsByTagName('bndbox')

    result = []
    for i in range(len(bnd_1)):
        xmin = int(bnd_1[i].childNodes[1].childNodes[0].nodeValue)
        ymin = int(bnd_1[i].childNodes[3].childNodes[0].nodeValue)
        xmax = int(bnd_1[i].childNodes[5].childNodes[0].nodeValue)
        ymax = int(bnd_1[i].childNodes[7].childNodes[0].nodeValue)
        result.append((xmin,ymin,xmax,ymax))
    
    # print(result)
    return result


def predict(img, model):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    predictions = model.predict(img, steps=1)
    score = np.argmax(predictions[0])
    return score


model_path = '/data/backup/pervinco_2020/datasets/total_beverage/walkin_beverage_final.h5'
model = tf.keras.models.load_model(model_path)
model.summary()

class_path = '/data/backup/pervinco_2020/datasets/total_beverage/labels_beverage_final.txt'
df = pd.read_csv(class_path, sep = ' ', index_col=False, header=None)
CLASS_NAMES = df[0].tolist()
CLASS_NAMES = sorted(CLASS_NAMES)
print(len(CLASS_NAMES))

main_boxes_path = '/data/backup/pervinco_2020/datasets/total_beverage/test.xml'
main_boxes = get_boxes(main_boxes_path)
print(main_boxes)

images_path = '/data/backup/pervinco_2020/datasets/total_beverage/0806(4)'
images = sorted(glob.glob(images_path + '/*'))

output_path = '/data/backup/pervinco_2020/datasets/total_beverage/seed_images/test'

for image in images:
    image = cv2.imread(image)
    
    idx = 0
    for coord in main_boxes:
        print(coord)
        xmin = int(coord[0])
        ymin = int(coord[1])
        xmax = int(coord[2])
        ymax = int(coord[3])

        cropped_img = image[ymin:ymax, xmin:xmax]
        score = predict(cropped_img, model)
        label = CLASS_NAMES[score]

        cropped_img2 = image[ymin+170:ymax, xmin:xmax]
        cropped_img3 = image[ymin:ymax, xmin+50:xmax]
        cropped_img4 = image[ymin+170:ymax, xmin+50:xmax]

        if not(os.path.isdir(output_path + '/crop_images/' + label)):
            os.makedirs(os.path.join(output_path + '/crop_images/' + label))

        else:
            pass
        
        now = datetime.datetime.now()
        nowdate = now.strftime('%Y-%m-%d %H:%M:%S')
        cv2.imwrite(output_path + '/crop_images/' + label + '/' + label + '_' + str(time.time()) + 'original' + str(idx) + '.jpg', cropped_img)
        cv2.imwrite(output_path + '/crop_images/' + label + '/' + label + '_' + str(time.time()) + '_hh_' + str(idx) + '.jpg', cropped_img2)
        cv2.imwrite(output_path + '/crop_images/' + label + '/' + label + '_' + str(time.time()) + '_ww_' + str(idx) + '.jpg', cropped_img3)
        cv2.imwrite(output_path + '/crop_images/' + label + '/' + label + '_' + str(time.time()) + '_hw_' + str(idx) + '.jpg', cropped_img4)
        idx+=1   