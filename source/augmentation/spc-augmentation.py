import os
import cv2
import random
import numpy as np
import pandas as pd
import albumentations as A

from tqdm import tqdm
from src.custom_aug import mosaic, mixup
from src.utils import read_label_file, read_xml, get_files, visualize, make_save_dir, write_xml

def data_process(is_train, folder_name):
    bg_files = []
    if INCLUDE_BG:
        ratio = int(BG_RATIO * len(annotations))

        for dir in BG_DIR:
            files = get_files(f"{dir}/images")
            files = random.sample(files, int(ratio / len(BG_DIR)))
            bg_files.extend(files)

    dataset = list(zip(images, annotations))

    if is_train:
        save_dir = f"{SAVE_DIR}/{folder_name}"
        make_save_dir(save_dir)

        for step in range(STEPS):
            random.shuffle(dataset)

            for idx in tqdm(range(len(annotations)), desc=f"STEP {step}"):
                image_path, annot_path = dataset[idx]
                opt = random.randint(0, 2)

                if opt == 0:
                    image, bboxes, labels = mosaic(idx, dataset, IMG_SIZE, classes)

                # elif opt == 1:
                #     image, bboxes, labels = mixup(idx, dataset, IMG_SIZE, classes, bg_files)

                else:
                    normal_transform = A.Compose([
                        A.Sequential([
                            A.Resize(IMG_SIZE, IMG_SIZE, p=1),
                            A.RandomBrightnessContrast(p=1, brightness_limit=(-0.2, 0.2)),

                            A.OneOf([
                                # A.Cutout(num_holes=32, max_h_size=16, max_w_size=16, fill_value=0, p=0.2),
                                A.Downscale(scale_min=0.5, scale_max=0.8, p=0.3),
                                A.RandomSnow(p=0.2),
                            ], p=0.5),
                        ])
                    ], bbox_params=A.BboxParams(format='pascal_voc', min_area=0.5, min_visibility=0.2, label_fields=['labels']))

                    image, annot = dataset[idx]
                    image = cv2.imread(image)
                    bboxes, labels = read_xml(annot, classes, 'pascal_voc')
                    transformed = normal_transform(image=image, bboxes=bboxes, labels=labels)
                    image, bboxes, labels = transformed['image'], transformed['bboxes'], transformed['labels']

                cv2.imwrite(f"{save_dir}/images/{folder_name}_{step}_{idx}.jpg", image)
                write_xml(f"{save_dir}/annotations", bboxes, labels, f"{folder_name}_{step}_{idx}", image.shape[0], image.shape[1], 'pascal_voc')
                
                if VISUAL:
                    print(opt)
                    visualize(image, bboxes, labels, 'pascal_voc', False)

        if INCLUDE_BG:
            for idx, file in enumerate(bg_files):
                bg_image = cv2.imread(file)
                bg_image = cv2.resize(bg_image, (IMG_SIZE, IMG_SIZE))
                cv2.imwrite(f"{save_dir}/images/bg_{idx}.jpg", bg_image)
                write_xml(f"{save_dir}/annotations", None, None, f"bg_{idx}", bg_image.shape[0], bg_image.shape[1], 'pascal_voc')

            print(f"Background Images {len(bg_files)} Added")

    else:
        save_dir = f"{SAVE_DIR}/{folder_name}"
        make_save_dir(save_dir)

        for idx in tqdm(range(len(annotations)), desc=f"valid"):
            image_path, annot_path = dataset[idx]

            valid_transform = A.Compose([
                A.Sequential([
                    A.Resize(IMG_SIZE, IMG_SIZE, p=1),
                    A.RandomBrightnessContrast(p=1, brightness_limit=(-0.2, 0.2)),
                ])
            ], bbox_params=A.BboxParams(format='pascal_voc', min_area=0.5, min_visibility=0.2, label_fields=['labels']))

            image, annot = dataset[idx]
            image = cv2.imread(image)
            bboxes, labels = read_xml(annot, classes, 'pascal_voc')
            transformed = valid_transform(image=image, bboxes=bboxes, labels=labels)
            image, bboxes, labels = transformed['image'], transformed['bboxes'], transformed['labels']

            cv2.imwrite(f"{save_dir}/images/{folder_name}_{idx}.jpg", image)
            write_xml(f"{save_dir}/annotations", bboxes, labels, f"{folder_name}_{idx}", image.shape[0], image.shape[1], 'pascal_voc')
            
            if VISUAL:
                print(opt)
                visualize(image, bboxes, labels, 'pascal_voc', False)


if __name__ == "__main__":
    ROOT_DIR = "/data/Datasets/SPC"
    FOLDER = "full-name6"
    STEPS = 1
    IMG_SIZE = 384
    BBOX_REMOVAL_THRESHOLD = 0.15
    VISUAL = False
    INCLUDE_BG = True
    BG_RATIO = 0.5
    BG_DIR = ["/data/Datasets/SPC/total-background"]
    
    IMG_DIR = f"{ROOT_DIR}/{FOLDER}/images"
    ANNOT_DIR = f"{ROOT_DIR}/{FOLDER}/annotations"
    LABEL_DIR = f"{ROOT_DIR}/Labels/labels.txt"
    SAVE_DIR = f"{ROOT_DIR}/{FOLDER}"

    classes = read_label_file(LABEL_DIR)
    images, annotations = get_files(IMG_DIR), get_files(ANNOT_DIR)
    
    data_process(True, "train")
    data_process(False, "valid")
