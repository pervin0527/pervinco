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

if __name__ == "__main__":
    root_dir = "/data/Models/yolov4"
    weight_file = f"{root_dir}/SPC/ckpt/full-name7/yolov4_last.weights"
    config_file = f"{root_dir}/SPC/deploy/yolov4.cfg"
    data_file = f"{root_dir}/SPC/data/spc.data"
    thresh_hold = .6

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 100)
    cap.set(cv2.CAP_PROP_FOURCC, 1196444237.0)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

    network, class_names, class_colors = darknet.load_network(config_file, data_file, weight_file, batch_size=1)
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)

    while True:
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
        image = cv2.resize(image, (960, 960))
        cv2.imshow("inference", image)

        key = cv2.waitKey(33)
        if key == 27 or key == 'esc':
            break

    if cap.isOpened():
        cap.release()

    cv2.destroyAllWindows()