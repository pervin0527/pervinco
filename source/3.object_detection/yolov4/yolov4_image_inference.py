from ctypes import *
import random
import os
import cv2
import time
import darknet
import argparse
from threading import Thread, enumerate
from queue import Queue

weight_file = "/data/Models/etri_yolov4/yolov4_final.weights"
config_file = "/home/barcelona/darknet/custom/etri/deploy/yolov4.cfg"
data_file = "/home/barcelona/darknet/custom/etri/data/etri.data"
thresh_hold = .4

network, class_names, class_colors = darknet.load_network(config_file, data_file, weight_file, batch_size=1)

width = darknet.network_width(network)
height = darknet.network_height(network)
darknet_image = darknet.make_image(width, height, 3)


test_image = cv2.imread("/data/Datasets/testset/ETRI_cropped_large/test_sample_07.jpg")
frame_rgb = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
frame_resized = cv2.resize(frame_rgb, (width, height), interpolation=cv2.INTER_LINEAR)
darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())

detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh_hold)
# darknet.print_detections(detections)

res = []
for i in range(len(detections)):
    res.append(detections[i][0])

image = darknet.draw_boxes(detections, frame_resized, class_colors)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# cv2.imwrite("/data/backup/pervinco_2020/darknet/build/darknet/x64/results/predictions.jpg", image)

print(res)
cv2.imshow("inference", image)
k = cv2.waitKey(0)