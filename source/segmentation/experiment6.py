""" Test : Calculate Class Weight"""
import cv2
import math
import numpy as np
import pandas as pd

from tqdm import tqdm
from PIL import Image
from glob import glob
from sklearn.utils.class_weight import compute_class_weight


def get_files(path):
    masks = sorted(glob(f"{path}/masks/*"))

    return masks, len(masks)


def create_class_weight(labels_dict,mu=0.15):
    total = np.sum(list(labels_dict.values()))
    keys = labels_dict.keys()
    class_weight = dict()
    
    for key in keys:
        score = math.log(mu*total/float(labels_dict[key]))
        class_weight[key] = score if score > 1.0 else 1.0
    
    return class_weight


if __name__ == "__main__":
    root_dir = "/data/Datasets/VOCdevkit/VOC2012"
    dataset_dir = f"{root_dir}/BASIC"
    label_dir = f"{root_dir}/Labels/class_labels.txt"
    height, width = 320, 320

    classes = pd.read_csv(label_dir, lineterminator='\n', header=None, index_col=False)
    classes = classes[0].to_list()
    colormap = [label for label in range(len(classes))]

    train_mask_files, total_train = get_files(f"{dataset_dir}/train")

    total_labels = set()
    pixels = [0] * len(classes)
    for idx in tqdm(range(len(train_mask_files))):
        mask = cv2.imread(train_mask_files[idx], cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (height, width))
        label, counts = np.unique(mask, return_counts=True)

        for idx, value in enumerate(counts):
            pixels[idx] += value
            total_labels.add(idx)

    print(total_labels)
    print(pixels)
    result = {}
    for key, value in zip(total_labels, pixels):
        result.update({key:value})

    class_weight = create_class_weight(result)
    print(class_weight)