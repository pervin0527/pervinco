import cv2
import os
import glob
import numpy as np

BASE_PATH = "/data/backup/pervinco_2020/tutorial/multi-class-bbox-regression/dataset"
IMAGES_PATH = os.path.join(BASE_PATH, "images")
ANNOTS_PATH = os.path.join(BASE_PATH, "annotations")

csv_files = sorted(glob.glob(ANNOTS_PATH + '/*.csv'))
for csv in csv_files:
    rows = open(csv).read().strip().split("\n")
    
    for row in rows:
        row = row.split(",")
        (filename, startX, startY, endX, endY, label) = row

        imagePath = IMAGES_PATH + '/' + label + '/' + filename
        image = cv2.imread(imagePath)
        # image = image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        (h, w) = image.shape[:2]
        print(h, w)

        cv2.rectangle(image, (int(startX), int(startY)), (int(endX), int(endY)), (0, 255, 0), 2)
        image = cv2.resize(image, (224, 224))

        cv2.imshow("data sample", image)
        cv2.waitKey(0)