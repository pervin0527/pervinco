import cv2
import numpy as np
import pandas as pd
from glob import glob
from PIL import Image

pixel_labels = [
    (0, 0, 0),
    (128, 0, 0),
    (0, 128, 0),
    (128, 128, 0),
    (0, 0, 128),
    (128, 0, 128),
    (0, 128, 128),
    (128, 128, 128),
    (64, 0, 0),
    (192, 0, 0),
    (64, 128, 0),
    (192, 128, 0),
    (64, 0, 128),
    (192, 0, 128),
    (64, 128, 128),
    (192, 128, 128),
    (0, 64, 0),
    (128, 64, 0),
    (0, 192, 0),
    (128, 192, 0),
    (0, 64, 128)
]

record = [0] * len(pixel_labels)

ds_path = "/data/Datasets/VOC_segmentation/train/annotations"
ds = sorted(glob(f"{ds_path}/*.png"))

for annotation in ds:
    img = Image.open(annotation)
    rgb_img = img.convert('RGB')
    width, height = img.size

    for x in range(width):
        for y in range(height):
            pixel_value = rgb_img.getpixel((x, y))

            if pixel_value != (0, 0, 0):
                if pixel_value in pixel_labels:
                    idx = pixel_labels.index(pixel_value)
                    record[idx] += 1

print(record)