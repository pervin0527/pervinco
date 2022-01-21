import random
import os
import cv2
import time
import darknet
import argparse
from tqdm import tqdm
from ctypes import *
from glob import glob
from queue import Queue
# from threading import Thread, enumerate

if __name__ == "__main__":
    ROOT_DIR = "/data/Models/yolov4"
    TEST_DIR = "/data/Datasets/SPC/full-name-test/images"
    WEIGHT_PATH = f"{ROOT_DIR}/SPC/ckpt/full-name7/yolov4_last.weights"
    CONFIG_PATH = f"{ROOT_DIR}/SPC/deploy/yolov4.cfg"
    DATA_PATH = f"{ROOT_DIR}/SPC/data/spc.data"
    THRESH_HOLD = .6
    VISUAL = False

    save_path = f"{('/').join(WEIGHT_PATH.split('/')[:-1])}/pred-result"
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    network, class_names, class_colors = darknet.load_network(CONFIG_PATH, DATA_PATH, WEIGHT_PATH, batch_size=1)

    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)

    test_images = glob(f'{TEST_DIR}/*')
    random.shuffle(test_images)
    print(len(test_images))

    for idx in tqdm(range(len(test_images))):
        test_image = test_images[idx]
        image = cv2.imread(test_image)
        frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height), interpolation=cv2.INTER_LINEAR)
        darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())

        detections = darknet.detect_image(network, class_names, darknet_image, thresh=THRESH_HOLD)
        # darknet.print_detections(detections)
        # print(detections)

        if detections:
            res = []
            for i in range(len(detections)):
                res.append(detections[i][0])

            image = darknet.draw_boxes(detections, frame_resized, class_colors)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if VISUAL:
            cv2.imshow("RESULT", image)
            cv2.waitKey(0)

        cv2.imwrite(f"{save_path}/result_{idx}.jpg", image)