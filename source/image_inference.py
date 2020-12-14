from ctypes import *
import glob
import math
import random
import os
import cv2
import numpy as np
import time
import darknet
import collections

def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def cvDrawBoxes(detections, img):
    labels = []
    for detection in detections:
        x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]
        xmin, ymin, xmax, ymax = convertBack(float(x), float(y), float(w), float(h))
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)

        if xmin >= 0 and ymin >= 0 and xmax >= 0 and ymax >= 0:
            cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
            cv2.putText(img,
                        detection[0].decode() +
                        " [" + str(round(detection[1] * 100, 2)) + "]",
                        (pt1[0], pt1[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        [255, 255, 0], 1)
            labels.append(detection[0].decode())

        else:
            pass
        
    print(collections.Counter(labels))
    
    return img


netMain = None
metaMain = None
altNames = None



def YOLO():
    global metaMain, netMain, altNames
    configPath = "./data/yolo-obj.cfg"
    weightPath = "./backup/yolo-obj_final.weights"
    metaPath= "./data/obj.data"
    img_path = '/home/barcelona/darknet/test_imgs/random'

    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath)+"`")
    if netMain is None:
        netMain = darknet.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass

    image_list = sorted(glob.glob(img_path + '/*.jpg'))
    print("num of Images :", len(image_list))

    darknet_image = darknet.make_image(darknet.network_width(netMain),
                                       darknet.network_height(netMain),
                                       3)
    for image in image_list:
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (darknet.network_width(netMain), darknet.network_height(netMain)), interpolation=cv2.INTER_NEAREST)
        darknet.copy_image_from_bytes(darknet_image,image.tobytes())

        detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.8)
        image = cvDrawBoxes(detections, image)
        image  = cv2.resize(image, (1920, 1080))
        # image = cv2.resize(image, (1280, 720))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        
        cv2.imshow('test', image)
        k = cv2.waitKey(0) & 0xFF

        if k == ord('q'):
            print("Finished")
            break

if __name__ == "__main__":
    YOLO()
