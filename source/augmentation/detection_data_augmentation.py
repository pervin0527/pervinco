import os
import cv2
import random
import numpy as np
import pandas as pd
import albumentations as A

from glob import glob
from tqdm import tqdm
from src.custom_aug import mosaic, mixup
from src.utils import read_label_file, read_xml, get_files, visualize, make_save_dir, write_xml

def background_process(save_dir):
    bg_files = []
    ratio = int(BG_RATIO * len(annotations))

    folders = sorted(glob(f"{BG_DIR}/*"))
    for folder in folders:
        files = glob(f"{folder}/*")
        random.shuffle(files)
        if len(files) > int(ratio / len(folders)):
            files = random.sample(files, int(ratio / len(folders)))

        print(folder, len(files))    
        bg_files.extend(files)        

    for idx, file in enumerate(bg_files):
        try:
            bg_image = cv2.imread(file)
            bg_image = cv2.resize(bg_image, (IMG_SIZE, IMG_SIZE))
            cv2.imwrite(f"{save_dir}/images/bg_{idx}.jpg", bg_image)
            write_xml(f"{save_dir}/annotations", None, None, f"bg_{idx}", bg_image.shape[0], bg_image.shape[1], 'pascal_voc')

        except:
            pass
    
    total = len(glob(f"{save_dir}/images/*"))
    print(f"Background Images {total} Added")

def data_process(is_train, folder_name):
    if is_train:
        rebuild_count = 0
        dataset = list(zip(images, annotations))
        save_dir = f"{SAVE_DIR}/{folder_name}"
        make_save_dir(save_dir)

        if INCLUDE_BG:
            background_process(save_dir)
            
        for step in range(STEPS):
            for idx in tqdm(range(len(dataset)), desc=f"Train {step}"):
                # print(len(dataset))
                opt = random.randint(0, 1)

                if opt == 0:
                    if len(dataset) < 4:
                        rebuild_count += 1
                        dataset.extend(list(zip(images, annotations)))
                        random.shuffle(dataset)
                    
                    pieces = random.sample(dataset, 4)
                    image, bboxes, labels = mosaic(pieces, IMG_SIZE, classes)
                    for piece in pieces:
                        target = dataset.index(piece)
                        dataset.pop(target)

                # elif opt == 1:
                #     image, bboxes, labels = mixup(idx, dataset, IMG_SIZE, classes, bg_files)

                else:
                    normal_transform = A.Compose([
                        A.Sequential([
                            A.Resize(IMG_SIZE, IMG_SIZE, p=1),
                            A.RandomBrightnessContrast(p=1, brightness_limit=(-0.2, 0.2)),
                            A.Downscale(scale_min=0.5, scale_max=0.8, p=0.3),
                            
                            # A.OneOf([
                            #     A.RandomRotate90(p=0.4),
                            #     A.Downscale(scale_min=0.5, scale_max=0.8, p=0.2),
                            #     A.VerticalFlip(p=0.2),
                            #     A.HorizontalFlip(p=0.2)
                            # ], p=1),

                            # A.OneOf([
                            #     A.Downscale(scale_min=0.5, scale_max=0.8, p=0.2),
                            #     # A.RandomSnow(p=0.2),
                            # ], p=1)

                        ])
                    ], bbox_params=A.BboxParams(format='pascal_voc', min_area=0.5, min_visibility=0.2, label_fields=['labels']))

                    if len(dataset) < 1:
                        rebuild_count += 1
                        dataset.extend(list(zip(images, annotations)))
                        random.shuffle(dataset)

                    item = random.sample(dataset, 1)[0]
                    image, annot = item
                    dataset.pop(dataset.index(item))
                    image = cv2.imread(image)
                    bboxes, labels = read_xml(annot, classes, 'pascal_voc')
                    transformed = normal_transform(image=image, bboxes=bboxes, labels=labels)
                    image, bboxes, labels = transformed['image'], transformed['bboxes'], transformed['labels']

                cv2.imwrite(f"{save_dir}/images/{folder_name}_{step}_{idx}.jpg", image)
                write_xml(f"{save_dir}/annotations", bboxes, labels, f"{folder_name}_{step}_{idx}", image.shape[0], image.shape[1], 'pascal_voc')
                
                if VISUAL:
                    # print(opt)
                    visualize(image, bboxes, labels, 'pascal_voc', False)

            print(f"Dataset Rebuilded {rebuild_count} times")
                
    else:
        dataset = list(zip(images, annotations))
        save_dir = f"{SAVE_DIR}/{folder_name}"
        make_save_dir(save_dir)

        for idx in tqdm(range(int(len(dataset) * VALID_RATIO)), desc=f"Valid 0"):
            # image_path, annot_path = dataset[idx]

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
                visualize(image, bboxes, labels, 'pascal_voc', False)


if __name__ == "__main__":
    ROOT_DIR = "/data/Datasets/SPC"
    FOLDER = "full-name13"
    STEPS = 1
    IMG_SIZE = 384
    VALID_RATIO = 0.1
    VISUAL = False
    INCLUDE_BG = True
    BG_RATIO = 0.2
    # BG_DIR = "/data/Datasets/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/TEST"
    BG_DIR = "/data/Datasets/SPC/download"
    
    IMG_DIR = f"{ROOT_DIR}/{FOLDER}/images"
    ANNOT_DIR = f"{ROOT_DIR}/{FOLDER}/annotations"
    LABEL_DIR = f"{ROOT_DIR}/Labels/labels.txt"
    SAVE_DIR = f"{ROOT_DIR}/{FOLDER}"

    classes = read_label_file(LABEL_DIR)
    images, annotations = get_files(IMG_DIR), get_files(ANNOT_DIR)
    
    data_process(True, "train")
    data_process(False, "valid")
