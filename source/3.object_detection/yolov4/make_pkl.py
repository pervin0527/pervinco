from ctypes import *
import random
import os
import cv2
import time
import darknet
import argparse
import pandas as pd
import numpy as np
from threading import Thread, enumerate
from queue import Queue

weight_file = "/home/barcelona/darknet/yolov4.weights"
config_file = "/home/barcelona/darknet/cfg/yolov4.cfg"
data_file = "/home/barcelona/darknet/cfg/coco.data"
thresh_hold = .4

network, class_names, class_colors = darknet.load_network(config_file, data_file, weight_file, batch_size=1)

width = darknet.network_width(network)
height = darknet.network_height(network)
darknet_image = darknet.make_image(width, height, 3)

server_root = '/home/ubuntu/metrabs/data/'
root = '/data/pose_estimation/data/'
files_txt = './3dhp_images_for_detection.txt'
output_path = f'{root}/3dhp/yolov4_person_detections.pkl'
files_list = pd.read_csv(files_txt, sep=' ', index_col=False, header=None)
files_list = sorted(files_list[0].tolist())

i = 0
detections_per_image = {}
for file in files_list:
    os.system('clear')
    print(len(files_list) - i)

    file = file.split('/')[5:]
    image_path = f'{root}' + '/'.join(file)
    # print(image_path)

    server_path = f'{server_root}' + '/'.join(file)
    # print(server_path)

    detections_per_image.setdefault(server_path, [])
    # detections_per_image.setdefault(image_path, [])
    # print(detections_per_image)

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)

    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh_hold)

    for idx in range(len(detections)):
        if detections[idx][0] == "person":
            confidence = float(detections[idx][1]) / 100
            x1, y1, w, h = int(detections[idx][2][0]), int(detections[idx][2][1]), int(detections[idx][2][2]), int(detections[idx][2][3])
            bbox = np.array([x1, y1, w, h])

            bbox_with_confidence = [bbox, confidence]
            detections_per_image[server_path].append(bbox_with_confidence)

    i += 1

n_images_without_detections = len([1 for x in detections_per_image.values() if not x])
n_detections = sum(len(v) for v in detections_per_image.values())

print(n_images_without_detections)
print(n_detections)

with open(out_path, 'wb') as f:
    pickle.dump(detections_per_image, f, protocol=pickle.HIGHEST_PROTOCOL)