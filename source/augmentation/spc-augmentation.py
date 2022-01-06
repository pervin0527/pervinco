import os
import cv2
import random
import numpy as np
import pandas as pd
import albumentations as A

from tqdm import tqdm
from src.utils import read_label_file, read_xml, get_files, visualize, make_save_dir, write_xml


def bb_overlap(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    iou = interArea / float(boxAArea)
    return iou


def refine_boxes(boxes):
    result_boxes = []
    for box in boxes:
        if box[2] - box[0] < 10 or box[3] - box[1] < 10:
            continue
        result_boxes.append(box)
    result_boxes = np.array(result_boxes)

    return result_boxes


def crop_image(image, boxes, labels, xmin, ymin, xmax, ymax):
    mosaic_transform = A.Compose([
        A.Resize(width=xmax-xmin, height=ymax-ymin, p=1),
        A.RandomBrightnessContrast(p=1, brightness_limit=(-0.2, 0.2)),

        # A.OneOf([
        #     # A.Cutout(num_holes=32, max_h_size=16, max_w_size=16, fill_value=0, p=0.2),
        #     A.Downscale(scale_min=0.5, scale_max=0.8, p=0.3),
        #     A.RandomSnow(p=0.2),
        # ], p=0.5),
    
    ], bbox_params=A.BboxParams(format='pascal_voc', min_area=0.2, min_visibility=0.2, label_fields=['labels']))
    transformed = mosaic_transform(image=image, bboxes=boxes, labels=labels)

    image, boxes, labels = transformed['image'], transformed['bboxes'], transformed['labels']
    result_boxes = np.array(boxes)

    return image, result_boxes


def mosaic(idx, ds):
    candidates = [idx]
    c = random.randint(0, len(ds))

    for r in range(3):
        while c in candidates:
            c = random.randint(0, len(ds))
        candidates.append(c)
    # print(candidates)
    
    result_image = np.full((IMG_SIZE, IMG_SIZE, 3), 1, dtype=np.uint8)
    result_boxes, result_labels = [], []

    xc, yc = [int(random.uniform(IMG_SIZE * 0.25, IMG_SIZE * 0.75)) for _ in range(2)]

    for i, id in enumerate(candidates):
        image, annot = ds[i]
        image = cv2.imread(image)
        bboxes, labels = read_xml(annot, classes, "pascal_voc")
        boxes = refine_boxes(bboxes)
        
        if i == 0 :
            image, boxes = crop_image(image, boxes, labels, IMG_SIZE-xc, IMG_SIZE-yc, IMG_SIZE, IMG_SIZE)
            if len(boxes) > 0:
                result_labels.extend(labels)

            result_image[0 : yc, 0 : xc, :] = image
            result_boxes.extend(boxes)

        elif i == 1:
            image, boxes, = crop_image(image, boxes, labels, 0, IMG_SIZE-yc, IMG_SIZE-xc, IMG_SIZE)
            if len(boxes) > 0:
                result_labels.extend(labels)

            result_image[0 : yc, xc : IMG_SIZE, :] = image

            if boxes.shape[0] > 0:
                boxes[:, [0, 2]] += xc

            result_boxes.extend(boxes)

        elif i == 2:
            image, boxes = crop_image(image, boxes, labels, 0, 0, IMG_SIZE-xc, IMG_SIZE-yc)
            if len(boxes) > 0:
                result_labels.extend(labels)

            result_image[yc:IMG_SIZE, xc:IMG_SIZE, :] = image
            if boxes.shape[0] > 0:
                boxes[:, [0, 2]] += xc
                boxes[:, [1, 3]] += yc

            result_boxes.extend(boxes)

        else:
            image, boxes = crop_image(image, boxes, labels, IMG_SIZE-xc, 0, IMG_SIZE, IMG_SIZE-yc)
            if len(boxes) > 0:
                result_labels.extend(labels)

            result_image[yc : IMG_SIZE, 0 : xc, :] = image
            if boxes.shape[0] > 0:
                boxes[ :, [1, 3]] += yc

            result_boxes.extend(boxes)

    # visualize(result_image, result_boxes, result_labels, 'pascal_voc', False)
    return result_image, result_boxes, result_labels


def mixup(idx, ds, noise_files, alpha=1.0):
    image, annot = ds[idx]
    image = cv2.imread(image)
    bboxes, labels = read_xml(annot, classes, 'pascal_voc')

    mixup_transform = A.Compose([
        A.Resize(width=IMG_SIZE, height=IMG_SIZE, p=1),
        A.RandomBrightnessContrast(p=1, brightness_limit=(-0.2, 0.2)),

        # A.OneOf([
        #     A.Cutout(num_holes=32, max_h_size=16, max_w_size=16, fill_value=0, p=0.2),
        #     A.Downscale(scale_min=0.5, scale_max=0.8, p=0.3),
        #     A.RandomSnow(p=0.2),
        # ], p=0.5),
    
    ], bbox_params=A.BboxParams(format='pascal_voc', min_area=0.2, min_visibility=0.2, label_fields=['labels']))
    transformed = mixup_transform(image=image, bboxes=bboxes, labels=labels)

    image, bboxes, labels = transformed['image'], transformed['bboxes'], transformed['labels']

    lam = np.clip(np.random.beta(alpha, alpha), 0, 0.2)

    rand_id = random.randint(0, len(noise_files)-1)
    noise_image = cv2.imread(noise_files[rand_id])
    noise_image = cv2.resize(noise_image, (IMG_SIZE, IMG_SIZE))
    mixedup_images = (lam*noise_image + (1 - lam)*image).astype(np.uint8)

    return mixedup_images, bboxes, labels


def data_process():
    make_save_dir(SAVE_DIR)

    bg_files = []
    if INCLUDE_BG:
        ratio = int(BG_RATIO * len(annotations))

        for dir in BG_DIR:
            files = get_files(f"{dir}/images")
            files = random.sample(files, int(ratio / len(BG_DIR)))
            bg_files.extend(files)

    # print(len(bg_files))

    for step in range(STEPS):
        dataset = list(zip(images, annotations))
        random.shuffle(dataset)

        for idx in tqdm(range(len(annotations)), desc=f"STEP {step}"):
            image_path, annot_path = dataset[idx]
            opt = random.randint(0, 2)

            if opt == 0:
                image, bboxes, labels = mosaic(idx, dataset)

            # elif opt == 1:
                # image, bboxes, labels = mixup(idx, dataset, bg_files)

            else:
                normal_transform = A.Compose([
                    A.Sequential([
                        A.Resize(IMG_SIZE, IMG_SIZE, p=1),
                        A.RandomBrightnessContrast(p=1, brightness_limit=(-0.2, 0.2)),

                        # A.OneOf([
                        #     # A.Cutout(num_holes=32, max_h_size=16, max_w_size=16, fill_value=0, p=0.2),
                        #     A.Downscale(scale_min=0.5, scale_max=0.8, p=0.3),
                        #     A.RandomSnow(p=0.2),
                        # ], p=0.5),
                    ])
                ], bbox_params=A.BboxParams(format='pascal_voc', min_area=0.5, min_visibility=0.2, label_fields=['labels']))

                image, annot = dataset[idx]
                image = cv2.imread(image)
                bboxes, labels = read_xml(annot, classes, 'pascal_voc')
                transformed = normal_transform(image=image, bboxes=bboxes, labels=labels)
                image, bboxes, labels = transformed['image'], transformed['bboxes'], transformed['labels']

            cv2.imwrite(f"{SAVE_DIR}/images/{FILE_NAME}_{step}_{idx}.jpg", image)
            write_xml(f"{SAVE_DIR}/annotations", bboxes, labels, f"{FILE_NAME}_{step}_{idx}", image.shape[0], image.shape[1], 'pascal_voc')
            
            if VISUAL:
                print(opt)
                visualize(image, bboxes, labels, 'pascal_voc', False)

    if INCLUDE_BG:
        for idx, file in enumerate(bg_files):
            bg_image = cv2.imread(file)
            bg_image = cv2.resize(bg_image, (IMG_SIZE, IMG_SIZE))
            cv2.imwrite(f"{SAVE_DIR}/images/bg_{idx}.jpg", bg_image)
            write_xml(f"{SAVE_DIR}/annotations", None, None, f"bg_{idx}", bg_image.shape[0], bg_image.shape[1], 'pascal_voc')


if __name__ == "__main__":
    ROOT_DIR = "/data/Datasets/SPC"
    FOLDER = "full-name2"
    STEPS = 1
    IMG_SIZE = 512
    BBOX_REMOVAL_THRESHOLD = 0.15
    VISUAL = False
    
    IMG_DIR = f"{ROOT_DIR}/{FOLDER}/images"
    ANNOT_DIR = f"{ROOT_DIR}/{FOLDER}/annotations"
    LABEL_DIR = f"{ROOT_DIR}/Labels/labels.txt"
    FILE_NAME = "valid"
    SAVE_DIR = f"{ROOT_DIR}/{FOLDER}/{FILE_NAME}"

    INCLUDE_BG = True
    BG_RATIO = 0.1
    BG_DIR = ["/data/Datasets/COCO2017", "/data/Datasets/SPC/Seeds/Background"]

    classes = read_label_file(LABEL_DIR)
    images, annotations = get_files(IMG_DIR), get_files(ANNOT_DIR)
    
    data_process()