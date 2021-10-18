from ctypes import *
import random
import os
import cv2
import time
import darknet
import argparse
from threading import Thread, enumerate
from queue import Queue
import datetime

weight_file = "/data/Models/DMC_yolov4/yolov4_last.weights"
config_file = "/home/barcelona/darknet/custom/DMC/deploy/yolov4.cfg"
data_file = "/home/barcelona/darknet/custom/DMC/data/dmc.data"
thresh_hold = .8

network, class_names, class_colors = darknet.load_network(config_file, data_file, weight_file, batch_size=1)

###############################################################################################
# cap = cv2.VideoCapture("/data/Datasets/Seeds/DMC/samples/sample_video_1.mp4")
cap = cv2.VideoCapture(0)
# print(cap.get(3), cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)

MJPG_CODEC = 1196444237.0 # MJPG
cap_AUTOFOCUS = 0
cap_FOCUS = 0
#cap_ZOOM = 400

frame_width = 540
frame_height = 960
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

cap.set(cv2.CAP_PROP_BRIGHTNESS, 100)
cap.set(cv2.CAP_PROP_FOURCC, MJPG_CODEC)
cap.set(cv2.CAP_PROP_AUTOFOCUS, cap_AUTOFOCUS)
cap.set(cv2.CAP_PROP_FOCUS, cap_FOCUS)
##############################################################################################
width = darknet.network_width(network)
height = darknet.network_height(network)
darknet_image = darknet.make_image(width, height, 3)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('/home/barcelona/test.avi', fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height), interpolation=cv2.INTER_LINEAR)
    darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())
    
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh_hold, hier_thresh=.5, nms=.45)
    # darknet.print_detections(detections)

    image = darknet.draw_boxes(detections, frame_resized, class_colors)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (frame_width, frame_height))
    cv2.imshow("inference", image)

    out.write(image)

    k = cv2.waitKey(1)
    if k == ord('q'):
        os.system('clear')
        break

out.release()
cap.release()
cv2.destroyAllWindows()