import os
import cv2
import random
import numpy as np
import albumentations as A
from glob import glob

from src.utils import read_xml

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
    # print(labels, boxes)
    if len(boxes) == len(labels):
        mosaic_transform = A.Compose([
            A.Resize(width=xmax-xmin, height=ymax-ymin, p=1),

            A.OneOf([
                A.OneOf([
                    A.RandomBrightnessContrast(brightness_limit=(-0.3, 0.3), contrast_limit=(-0.3, 0.3), p=0.5),
                    A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=(0, 0), val_shift_limit=(0, 100), p=0.5),
                ], p=1),

                A.OneOf([
                    A.Downscale(scale_min=0.9, scale_max=0.95, p=0.3),
                    A.MotionBlur(blur_limit=(3, 4), p=0.3)
                ], p=0.3)
            ], p=1),
        
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
        transformed = mosaic_transform(image=image, bboxes=boxes, labels=labels)

        image, boxes, labels = transformed['image'], transformed['bboxes'], transformed['labels']
        result_boxes = np.array(boxes)

    else:
        image = np.zeros(shape=((ymax-ymin), (xmax-xmin), 3), dtype=np.uint8)
        result_boxes = np.array([])
    
    return image, result_boxes


def mosaic(pieces, img_size, classes):
    result_image = np.full((img_size, img_size, 3), 1, dtype=np.uint8)
    result_boxes, result_labels = [], []
    xc, yc = [int(random.uniform(img_size * 0.25, img_size * 0.75)) for _ in range(2)]
    for i, piece in enumerate(pieces):
        image, annot = piece
        # print(image, annot)
        image = cv2.imread(image)
        bboxes, labels = read_xml(annot, classes, "pascal_voc")
        boxes = refine_boxes(bboxes)
        
        if i == 0 :
            image, boxes = crop_image(image, boxes, labels, img_size-xc, img_size-yc, img_size, img_size)
            if len(boxes) > 0:
                result_labels.extend(labels)

            result_image[0 : yc, 0 : xc, :] = image
            result_boxes.extend(boxes)

        elif i == 1:
            image, boxes, = crop_image(image, boxes, labels, 0, img_size-yc, img_size-xc, img_size)
            if len(boxes) > 0:
                result_labels.extend(labels)

            result_image[0 : yc, xc : img_size, :] = image

            if boxes.shape[0] > 0:
                boxes[:, [0, 2]] += xc

            result_boxes.extend(boxes)

        elif i == 2:
            image, boxes = crop_image(image, boxes, labels, 0, 0, img_size-xc, img_size-yc)
            if len(boxes) > 0:
                result_labels.extend(labels)

            result_image[yc:img_size, xc:img_size, :] = image
            if boxes.shape[0] > 0:
                boxes[:, [0, 2]] += xc
                boxes[:, [1, 3]] += yc

            result_boxes.extend(boxes)

        else:
            image, boxes = crop_image(image, boxes, labels, img_size-xc, 0, img_size, img_size-yc)
            if len(boxes) > 0:
                result_labels.extend(labels)

            result_image[yc : img_size, 0 : xc, :] = image
            if boxes.shape[0] > 0:
                boxes[ :, [1, 3]] += yc

            result_boxes.extend(boxes)

    # visualize(result_image, result_boxes, result_labels, 'pascal_voc', False)
    return result_image, result_boxes, result_labels


def mixup(image, bboxes, labels, img_size, mixup_bg, min=0.4, max=0.5, alpha=1.0):
    main_transform = A.Compose([
        A.Resize(width=img_size, height=img_size, p=1),
        A.RandomBrightnessContrast(p=1, brightness_limit=(-0.3, 0.2)),
    ], bbox_params=A.BboxParams(format='pascal_voc', min_area=0.2, min_visibility=0.2, label_fields=['labels']))
    transformed = main_transform(image=image, bboxes=bboxes, labels=labels)

    image, bboxes, labels = transformed['image'], transformed['bboxes'], transformed['labels']

    lam = np.clip(np.random.beta(alpha, alpha), min, max)

    background_transform = A.Compose([
        A.Resize(width=img_size, height=img_size, p=1),

        A.OneOf([
            A.RandomRotate90(p=0.3),
            A.HorizontalFlip(p=0.3),
            A.VerticalFlip(p=0.3),
        ], p=1),

        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=(-0.4, 0.4), p=0.3),
            A.HueSaturationValue(val_shift_limit=(40, 80), p=0.3),
            # A.ChannelShuffle(p=0.3)
        ], p=1)
    ])

    bg_files = glob(f"{mixup_bg}/*")
    rand_id = random.randint(0, len(bg_files)-1)
    background_image = cv2.imread(bg_files[rand_id])
    transformed = background_transform(image=background_image)
    background_image = transformed['image']
    
    mixedup_images = (lam*background_image + (1 - lam)*image).astype(np.uint8)

    return mixedup_images, bboxes, labels