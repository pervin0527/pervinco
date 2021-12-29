import os
import cv2
import random
import pathlib
import numpy as np
import pandas as pd
import albumentations as A
from tqdm import tqdm
from src.custom_aug import MixUp, CutMix, Mosaic, CustomCutout
from src.data import BaseDataset, LoadImages, LoadPascalVOCLabels, Augmentations
from src.utils import read_label_file, read_xml, get_files, write_xml, make_save_dir, visualize

def mixup(input_image, noise_files, alpha=1.0):
    lam = np.clip(np.random.beta(alpha, alpha), 0.1, 0.2)

    rand_id = random.randint(0, len(noise_files))
    noise_image = cv2.imread(noise_files[rand_id])
    noise_image = cv2.resize(noise_image, (IMG_SIZE, IMG_SIZE))
    mixedup_images = (lam*noise_image + (1 - lam)*input_image).astype(np.uint8)

    return mixedup_images

if __name__ == "__main__":
    ROOT_DIR = "/data/Datasets/SPC/full-name-test"
    LABEL_DIR = "/data/Datasets/SPC/Labels/labels.txt"
    SAVE_DIR = f"{ROOT_DIR}/test"
    dataset_name = "test"
    EPOCH = 1
    IMG_SIZE = 384
    VISUAL = True
    
    IMG_DIR = f"{ROOT_DIR}/images"
    ANNOT_DIR = f"{ROOT_DIR}/annotations"
    classes = read_label_file(LABEL_DIR)
    make_save_dir(SAVE_DIR)

    NOISE_DIR = "/data/Datasets/COCO2017/images"
    noise_files = get_files(NOISE_DIR)

    images, annotations = get_files(IMG_DIR), get_files(ANNOT_DIR)
    print(classes)
    print(len(images), len(annotations))
    dataset = BaseDataset(images, annotations, classes)
    dataset = LoadImages(dataset)
    dataset = LoadPascalVOCLabels(dataset)

    transform = A.Compose([
        A.OneOf([
            A.Sequential([
                A.RandomBrightnessContrast(p=1),
                A.RGBShift(p=1, r_shift_limit=(-10, 10), g_shift_limit=(-10, 10), b_shift_limit=(-10, 10)),
                A.Downscale(scale_min=0.3, scale_max=0.8, p=0.5),
                A.RandomSnow(p=0.5),
            ]),

            A.Sequential([
                Mosaic(
                    dataset,
                    transforms=[
                        A.RandomBrightnessContrast(p=1),
                        A.RGBShift(p=1, r_shift_limit=(-10, 10), g_shift_limit=(-10, 10), b_shift_limit=(-10, 10)),
                        A.Downscale(scale_min=0.3, scale_max=0.8, p=0.5),
                    ],
                    always_apply=True
                ),
            ]),

        ], p=1),
        
        A.Cutout(num_holes=128, max_h_size=32, max_w_size=32, fill_value=0, p=.4),
        A.Resize(height=IMG_SIZE, width=IMG_SIZE, p=1),
    ], bbox_params=A.BboxParams(format=dataset.bbox_format, min_area=0.5, min_visibility=0.2, label_fields=['labels']))

    transformed = Augmentations(dataset, transform)
    length = transformed.__len__()

    make_save_dir(SAVE_DIR)
    for ep in range(EPOCH):
        indexes = list(range(length))
        random.shuffle(indexes)

        for i in tqdm(range(length), desc=f"epoch {ep}"):
            i = indexes[i]
            file_no = ep*length+i

            try:
                output = transformed[i]

                if random.random() > 0.5:
                    output['image'] = mixup(output['image'], noise_files)

                if len(output['bboxes']) > 0:
                    cv2.imwrite(f'{SAVE_DIR}/images/{dataset_name}_{ep}_{file_no}.jpg', output['image'])
                    height, width = output['image'].shape[:-1]
                    write_xml(f"{SAVE_DIR}/annotations", output['bboxes'], output['labels'], f'{dataset_name}_{ep}_{file_no}', height, width, 'albumentations')

                    if VISUAL:
                        visualize(output['image'], output['bboxes'], output['labels'], format='albumentations', show_info=True)

            except:
                pass