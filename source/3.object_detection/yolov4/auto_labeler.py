from ctypes import *
import os
import cv2
import time
import darknet
import argparse
import pathlib
import pprint

from queue import Queue
from threading import Thread, enumerate
from xml.dom.minidom import parseString
from xml.etree.ElementTree import ElementTree
from lxml.etree import Element, SubElement, tostring
from sklearn.model_selection import train_test_split

def rewrite_xml(results, is_train, idx):
    if is_train:
        path = "train"

    else:
        path = "test"

    node_root = Element('annotation')
    
    node_folder = SubElement(node_root, 'folder')
    node_folder.text = path
    
    node_filename = SubElement(node_root, 'filename')
    node_filename.text = f'image_{idx}.jpg'
    
    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = str(width)
    
    node_height = SubElement(node_size, 'height')
    node_height.text = str(height)
    
    node_depth = SubElement(node_size, 'depth')
    node_depth.text = '3'
    
    for i in range(len(results)):
        node_object = SubElement(node_root, 'object')
        node_name = SubElement(node_object, 'name')
        node_name.text = str(results[i][0])

        node_pose = SubElement(node_object, 'pose')
        node_pose.text = 'Unspecified'

        node_truncated = SubElement(node_object, 'truncated')
        node_truncated.text = '0'

        node_occluded = SubElement(node_object, 'occluded')
        node_occluded.text = '0'

        node_difficult = SubElement(node_object, 'difficult')
        node_difficult.text = '0'

        node_bndbox = SubElement(node_object, 'bndbox')
        node_xmin = SubElement(node_bndbox, 'xmin')
        node_xmin.text = str(results[i][1])
        node_ymin = SubElement(node_bndbox, 'ymin')
        node_ymin.text = str(results[i][2])
        node_xmax = SubElement(node_bndbox, 'xmax')
        node_xmax.text = str(results[i][3])
        node_ymax = SubElement(node_bndbox, 'ymax')
        node_ymax.text = str(results[i][4])
        
    tree = ElementTree(node_root)
    tree.write(f'/data/Datasets/Seeds/mm_etri/{path}/image_{idx}.xml')


def save_detection_result(images, is_train):
    if is_train:
        path = "train"

    else:
        path = "test"

    idx = 0
    for image in images:
        test_image = cv2.imread(image)
        frame_rgb = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height))
        darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())

        detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh_hold)
        # print(detections)

        result = []
        for i in range(len(detections)):
            class_name = detections[i][0]
            x = detections[i][2][0]
            y = detections[i][2][1]
            w = detections[i][2][2]
            h = detections[i][2][3]

            xmin, ymin, xmax, ymax = darknet.bbox2points((x, y, w, h))
            result.append([class_name, xmin, ymin, xmax, ymax])

        frame_resized = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f"/data/Datasets/Seeds/mm_etri/{path}/image_{idx}.jpg", frame_resized)
        print(result)

        rewrite_xml(result, is_train, idx)
        idx += 1


if __name__ == "__main__":
    weight_file = "/home/barcelona/darknet/custom/fire/ckpt/21_07_09/yolov4_final.weights"
    config_file = "/home/barcelona/darknet/custom/fire/deploy/yolov4.cfg"
    data_file = "/home/barcelona/darknet/custom/fire/data/fire.data"
    thresh_hold = .4

    network, class_names, class_colors = darknet.load_network(config_file, data_file, weight_file, batch_size=1)

    image_path = "/data/Datasets/Seeds/ETRI_detection/images"
    images = pathlib.Path(image_path)
    images = list(images.glob('*.jpg'))
    images = sorted([str(path) for path in images])

    train_images, test_images = train_test_split(images, test_size=0.2, shuffle=True)

    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)

    save_detection_result(train_images, True)
    save_detection_result(test_images, False)