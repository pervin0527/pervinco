import os
import cv2
import random
import pathlib
import pandas as pd
import albumentations as A
from tqdm import tqdm
from src.custom_aug import MixUp, CutMix, Mosaic
from src.data import BaseDataset, LoadImages, LoadPascalVOCLabels, Augmentations
from src.utils import read_label_file, read_xml, get_files, write_xml, make_save_dir, visualize

if __name__ == "__main__":
    ROOT_DIR = "/data/Datasets/SPC/full-name1"
    LABEL_DIR = "/data/Datasets/SPC/Labels/labels.txt"
    SAVE_DIR = "/data/Datasets/SPC/full-name1/augmentations"
    EPOCH = 4
    
    mix_bg = True
    bg_ratio = 0.1
    bg_dir = "/data/Datasets/COCO2017/images"

    IMG_DIR = f"{ROOT_DIR}/images"
    ANNOT_DIR = f"{ROOT_DIR}/annotations"
    images, annotations = get_files(IMG_DIR), get_files(ANNOT_DIR)
    classes = read_label_file(LABEL_DIR)

    print(classes)
    print(len(images), len(annotations))
    dataset = BaseDataset(images, annotations, classes)
    dataset = LoadImages(dataset)
    dataset = LoadPascalVOCLabels(dataset)

    transform = A.Compose([
    A.OneOf([
        A.Sequential([
            A.Rotate(limit=5, p=1, border_mode=0),
            MixUp(dataset, rate_range=(0, 0.1), mix_label=False, p=0.5),
            A.RandomBrightnessContrast(p=1),
            A.RGBShift(p=1, r_shift_limit=(-10, 10), g_shift_limit=(-10, 10), b_shift_limit=(-10, 10)),
            A.ISONoise(p=0.5)
        ]),
        A.Sequential([
            Mosaic(
                dataset,
                transforms=[
                    A.Rotate(limit=5, p=1, border_mode=0),
                    MixUp(dataset, rate_range=(0, 0.1), mix_label=False, p=0.5),
                    A.RandomBrightnessContrast(p=1),
                    A.RGBShift(p=1, r_shift_limit=(-10, 10), g_shift_limit=(-10, 10), b_shift_limit=(-10, 10)),
                    A.ISONoise(p=0.5)
                ],
                always_apply=True
            ),
        ])
    ], p=1),
    
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
                visualize(output['image'], output['bboxes'], output['labels'], format="albumentations")
                cv2.imwrite(f'{SAVE_DIR}/images/{file_no}.jpg', output['image'])
                height, width = output['image'].shape[:-1]
                write_xml(f"{SAVE_DIR}/annotations", output['bboxes'], output['labels'], file_no, height, width, 'albumentations')

            except:
                pass

    if mix_bg:
        bg_images = random.sample(get_files(bg_dir), int(length*bg_ratio))
        for bg_img in bg_images:
            file_name = bg_img.split('/')[-1].split('.'[0])
            image = cv2.imread(bg_img)
            height, width = image.shape[:-1]
            cv2.imwrite(f"{SAVE_DIR}/images/bg_{file_name}.jpg", image)
            write_xml(f"{SAVE_DIR}/annotations", None, None, f'bg_{file_name}', height, width, None)
