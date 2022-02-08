import os
import cv2
import time
import random
import darknet
import argparse
from ctypes import *
from queue import Queue
# from threading import Thread, enumerate
from yolov4_custom_utils import output_remake, write_xml

if __name__ == "__main__":
    ROOT_DIR = "/data/Models/yolov4/SPC"
    WEIGHT_PATH = f"{ROOT_DIR}/ckpt/recal_anchor/yolov4_final.weights"
    CONFIG_PATH = f"{ROOT_DIR}/deploy/yolov4.cfg"
    DATA_PATH = f"{ROOT_DIR}/data/spc.data"
    THRESH_HOLD = .7

    VISUAL = True
    SAVE_RESULT = True
    VIDEO_PATH = "/data/Datasets/SPC/Videos/2021-11-24/20211124_142829.mp4"
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 100)
    cap.set(cv2.CAP_PROP_FOURCC, 1196444237.0)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

    if SAVE_RESULT:
        folder_name = VIDEO_PATH.split('/')[-1].split('.')[0]
        SAVE_PATH = f"{('/').join(WEIGHT_PATH.split('/')[:-1])}/labels/{folder_name}"
        if not os.path.isdir(SAVE_PATH):
            os.makedirs(f"{SAVE_PATH}/images")
            os.makedirs(f"{SAVE_PATH}/annotations")
            os.makedirs(f"{SAVE_PATH}/result")

    network, class_names, class_colors = darknet.load_network(CONFIG_PATH, DATA_PATH, WEIGHT_PATH, batch_size=1)
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height), interpolation=cv2.INTER_LINEAR)
        copy_resized = cv2.cvtColor(frame_resized.copy(), cv2.COLOR_BGR2RGB)
        darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())
        
        detections = darknet.detect_image(network, class_names, darknet_image, thresh=THRESH_HOLD, hier_thresh=.5, nms=.45)
        # darknet.print_detections(detections)
        result_image = darknet.draw_boxes(detections, frame_resized, class_colors)
        result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)

        if VISUAL:
            cv2.imshow("inference", result_image)
            key = cv2.waitKey(33)
            if key == 27 or key == 'esc':
                break

        if SAVE_RESULT:
            labels, scores, bboxes = output_remake(detections)
            write_xml(f"{SAVE_PATH}/annotations", bboxes, labels, f"FRAME_{idx:>06}", frame_resized.shape)
            cv2.imwrite(f"{SAVE_PATH}/images/FRAME_{idx:>06}.jpg", copy_resized)
            cv2.imwrite(f"{SAVE_PATH}/result/FRAME_{idx:>06}.jpg", result_image)
            idx += 1

    if cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()