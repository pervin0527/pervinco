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
    ROOT_DIR = "/data/Datasets/SPC/full-name2"
    LABEL_DIR = "/data/Datasets/SPC/Labels/labels.txt"
    SAVE_DIR = f"{ROOT_DIR}/train2"
    dataset_name = "main2"
    EPOCH = 3
    IMG_SIZE = 384
    VISUAL = False
    
    IMG_DIR = f"{ROOT_DIR}/images"
    ANNOT_DIR = f"{ROOT_DIR}/annotations"
    images, annotations = get_files(IMG_DIR), get_files(ANNOT_DIR)
    classes = read_label_file(LABEL_DIR)

    INCLUDE_BG = False
    BG_RATIO = 0.1
    BG_DIR = ["/data/Datasets/COCO2017", "/data/Datasets/SPC/Seeds/Background"]

    make_save_dir(SAVE_DIR)
    images, annotations = get_files(IMG_DIR), get_files(ANNOT_DIR)
    classes = read_label_file(LABEL_DIR)

    if INCLUDE_BG:
        for bg in BG_DIR:
            bg_images = get_files(f"{bg}/images")
            bg_images = random.sample(bg_images, int(len(images) * BG_RATIO))
            print(len(bg_images))

            for bg_img in bg_images:
                filename = bg_img.split('/')[-1].split('.')[0]
                image = cv2.imread(bg_img)
                height, width = image.shape[:-1]
                cv2.imwrite(f"{SAVE_DIR}/images/bg_{filename}.jpg", image)
                write_xml(f"{SAVE_DIR}/annotations", None, None, f"bg_{filename}", height, width, 'pascal_voc')

    print(classes)
    print(len(images), len(annotations))
    dataset = BaseDataset(images, annotations, classes)
    dataset = LoadImages(dataset)
    dataset = LoadPascalVOCLabels(dataset)

    transform = A.Compose([
        A.OneOf([
            A.Sequential([
                A.Resize(height=IMG_SIZE, width=IMG_SIZE, p=1),
                A.Rotate(limit=5, p=1, border_mode=0),
                MixUp(dataset, rate_range=(0, 0.1), mix_label=False, p=0.5),
                A.RandomBrightnessContrast(p=1),
                A.RGBShift(p=1, r_shift_limit=(-10, 10), g_shift_limit=(-10, 10), b_shift_limit=(-10, 10)),
                A.ISONoise(p=0.5),
            ]),

            A.Sequential([
                Mosaic(
                    dataset,
                    transforms=[
                        A.Rotate(limit=5, p=1, border_mode=0),
                        MixUp(dataset, rate_range=(0, 0.1), mix_label=False, p=0.5),
                        A.RandomBrightnessContrast(p=1),
                        A.RGBShift(p=1, r_shift_limit=(-10, 10), g_shift_limit=(-10, 10), b_shift_limit=(-10, 10)),
                        A.ISONoise(p=0.5),
                    ],
                    always_apply=True
                ),
                A.Resize(height=IMG_SIZE, width=IMG_SIZE, p=1),
            ]),

            # A.Sequential([
            #     A.ShiftScaleRotate(border_mode=1, rotate_limit=(0), scale_limit=(0, 0), shift_limit=(-0.35, 0.35)),
            #     A.RandomSizedCrop([IMG_SIZE, 1440], IMG_SIZE, IMG_SIZE, p=1),
            #     MixUp(dataset, rate_range=(0, 0.05), mix_label=False, p=0.5),
            #     A.RandomBrightnessContrast(p=1),
            #     A.RGBShift(p=1, r_shift_limit=(-10, 10), g_shift_limit=(-10, 10), b_shift_limit=(-10, 10)),
            #     A.ISONoise(p=0.5)
            # ])
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

                if len(output['bboxes']) > 0:
                    cv2.imwrite(f'{SAVE_DIR}/images/{dataset_name}_{ep}_{file_no}.jpg', output['image'])
                    height, width = output['image'].shape[:-1]
                    write_xml(f"{SAVE_DIR}/annotations", output['bboxes'], output['labels'], f'{dataset_name}_{ep}_{file_no}', height, width, 'albumentations')

                    if VISUAL:
                        visualize(output['image'], output['bboxes'], output['labels'], format='albumentations', show_info=True)

            except:
                pass