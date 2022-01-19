import random
import os
import cv2
import time
import darknet
import argparse
from ctypes import *
from glob import glob
from queue import Queue
# from threading import Thread, enumerate

testset_path = "/home/ubuntu/Datasets/SPC/full-name7/valid/images"
weight_file = "/home/ubuntu/Models/yolov4-spc/yolov4_last.weights"
config_file = "/home/ubuntu/darknet/custom/SPC/deploy/yolov4.cfg"
data_file = "/home/ubuntu/darknet/custom/SPC/data/spc.data"
thresh_hold = .6

save_path = f"{('/').join(weight_file.split('/')[:-1])}/pred-result"
if not os.path.isdir(save_path):
    os.makedirs(save_path)

network, class_names, class_colors = darknet.load_network(config_file, data_file, weight_file, batch_size=1)

width = darknet.network_width(network)
height = darknet.network_height(network)
darknet_image = darknet.make_image(width, height, 3)

test_images = glob(f'{testset_path}/*')
random.shuffle(test_images)
print(len(test_images))

for idx, test_image in enumerate(test_images):
    print(idx)
    test_image = cv2.imread(test_image)
    frame_rgb = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height), interpolation=cv2.INTER_LINEAR)
    darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())

    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh_hold)
    # darknet.print_detections(detections)
    # print(detections)

    if detections:
        res = []
        for i in range(len(detections)):
            res.append(detections[i][0])

        image = darknet.draw_boxes(detections, frame_resized, class_colors)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    cv2.imwrite(f"{save_path}/result_{idx}.jpg", image)

    if idx == 100:
        break