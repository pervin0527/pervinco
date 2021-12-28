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
    ROOT_DIR = "/data/Datasets/SPC/full-name3/train"
    LABEL_DIR = "/data/Datasets/SPC/Labels/labels.txt"
    SAVE_DIR = f"{ROOT_DIR}/augmentations"
    dataset_name = "main"
    EPOCH = 10
    IMG_SIZE = 384
    VISUAL = False
    
    IMG_DIR = f"{ROOT_DIR}/images"
    ANNOT_DIR = f"{ROOT_DIR}/annotations"
    classes = read_label_file(LABEL_DIR)
    make_save_dir(SAVE_DIR)

    INCLUDE_BG = True
    BG_RATIO = 0.1
    BG_DIR = ["/data/Datasets/COCO2017", "/data/Datasets/SPC/Seeds/Background"]

    seeds = len(get_files(IMG_DIR))
    if INCLUDE_BG:
        for bg in BG_DIR:
            bg_images = get_files(f"{bg}/images")
            bg_images = random.sample(bg_images, int(seeds * BG_RATIO))
            print(len(bg_images))

            for bg_img in bg_images:
                filename = bg_img.split('/')[-1].split('.')[0]
                image = cv2.imread(bg_img)
                height, width = image.shape[:-1]
                cv2.imwrite(f"{IMG_DIR}/bg_{filename}.jpg", image)
                write_xml(f"{ANNOT_DIR}", None, None, f"bg_{filename}", height, width, 'pascal_voc')

    images, annotations = get_files(IMG_DIR), get_files(ANNOT_DIR)
    print(classes)
    print(len(images), len(annotations))
    dataset = BaseDataset(images, annotations, classes)
    dataset = LoadImages(dataset)
    dataset = LoadPascalVOCLabels(dataset)

    transform = A.Compose([
        A.Sequential([
            MixUp(dataset, rate_range=(0.1, 0.2), mix_label=False, p=1),
            A.RandomBrightnessContrast(p=1),
        ]),

        A.OneOf([
            A.Downscale(scale_min=0.3, scale_max=0.8, p=1),
            A.MotionBlur(blur_limit=(3, 7), p=1)
        ], p=1),

        A.Resize(IMG_SIZE, IMG_SIZE, p=1),

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