""" Test : Calculate Class Weight"""
import cv2
import math
import numpy as np
import pandas as pd

from tqdm import tqdm
from PIL import Image
from glob import glob
from sklearn.utils.class_weight import compute_class_weight

def get_dataset_counts(d):
    pixel_count = np.array([i for i in d.values()])

    sum_pixel_count = 0
    for i in pixel_count:
        sum_pixel_count += i

    return pixel_count, sum_pixel_count

def get_dataset_statistics(pixel_count, sum_pixel_count):
    
    pixel_frequency = np.round(pixel_count / sum_pixel_count, 4)

    mean_pixel_frequency = np.round(np.mean(pixel_frequency), 4)

    return pixel_frequency, mean_pixel_frequency

def get_balancing_class_weights(classes, d):
    CLASSES = classes
    pixel_count, sum_pixel_count = get_dataset_counts(d)

    background_pixel_count = 0
    mod_pixel_count = []

    for c in CLASSES:
        if c not in classes:
            background_pixel_count += d[c]
        else:
            mod_pixel_count.append(d[c])

    # if not ALL_CLASSES:
    #     mod_pixel_count.append(background_pixel_count)
    # else:
    #     mod_pixel_count[:-1]
    
    pixel_frequency, mean_pixel_frequency = get_dataset_statistics(mod_pixel_count, sum_pixel_count)

    class_weights = np.round(mean_pixel_frequency / pixel_frequency, 2)
    return class_weights    


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


def analyze_dataset(masks, classes, height, width):
    labels_check = set()
    class_per_pixels = {key : 0 for key in range(len(classes))}

    for idx in tqdm(range(len(masks))):
        mask = cv2.imread(masks[idx], cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (height, width))
        label, counts = np.unique(mask, return_counts=True)
        # print(label, counts)

        for i in range(len(label)):
            labels_check.add(label[i])
            class_per_pixels[label[i]] += counts[i]

    return class_per_pixels


if __name__ == "__main__":
    root_dir = "/data/Datasets/VOCdevkit/VOC2012"
    dataset_dir = f"{root_dir}/BASIC"
    label_dir = f"{root_dir}/Labels/class_labels.txt"
    height, width = 320, 320

    classes = pd.read_csv(label_dir, lineterminator='\n', header=None, index_col=False)
    classes = classes[0].to_list()
    colormap = [label for label in range(len(classes))]

    train_mask_files, total_train = get_files(f"{dataset_dir}/train")

    class_per_pixels = analyze_dataset(train_mask_files, classes, height, width)
    print(class_per_pixels)
    print()
    
    class_weights = create_class_weight(class_per_pixels)
    print(list(class_weights.values()))