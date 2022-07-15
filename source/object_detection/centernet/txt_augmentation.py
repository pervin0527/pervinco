import os
import cv2
import random
import numpy as np
import albumentations as A

from tqdm import tqdm
from glob import glob


def augmentation(data_list):
    image, bboxes, classes = data_list[0]
    image = cv2.imread(image)
    transformed = transform(image=image, bboxes=bboxes, labels=classes)
    transformed_image, transformed_bboxes, transformed_labels = transformed["image"], transformed["bboxes"], transformed["labels"]

    return transformed_image, transformed_bboxes, transformed_labels


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
                    A.RandomBrightnessContrast(brightness_limit=(-0.25, 0.25), contrast_limit=(-0.25, 0.25), p=0.5),
                    A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=(0, 0), val_shift_limit=(0, 100), p=0.5),
                ], p=1),

                A.OneOf([
                    A.Downscale(scale_min=0.9, scale_max=0.95, p=0.5),
                    A.MotionBlur(blur_limit=(3, 4), p=0.5)
                ], p=0.4)

            ], p=1),
        
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
        transformed = mosaic_transform(image=image, bboxes=boxes, labels=labels)

        image, boxes, labels = transformed['image'], transformed['bboxes'], transformed['labels']
        result_boxes = np.array(boxes)

    else:
        image = np.zeros(shape=((ymax-ymin), (xmax-xmin), 3), dtype=np.uint8)
        result_boxes = np.array([])
    
    return image, result_boxes


def mosaic(data_list):
    result_image = np.full((512, 512, 3), 1, dtype=np.uint8)
    result_boxes, result_labels = [], []
    xc, yc = [int(np.random.uniform(512 * 0.25, 512 * 0.75)) for _ in range(2)]

    for i, data in enumerate(data_list):
        image, bboxes, labels = data
        image = cv2.imread(image)
        boxes = refine_boxes(bboxes)
        
        if i == 0 :
            image, boxes = crop_image(image, boxes, labels, 512-xc, 512-yc, 512, 512)
            if len(boxes) > 0:
                result_labels.extend(labels)

            result_image[0 : yc, 0 : xc, :] = image
            result_boxes.extend(boxes)

        elif i == 1:
            image, boxes, = crop_image(image, boxes, labels, 0, 512-yc, 512-xc, 512)
            if len(boxes) > 0:
                result_labels.extend(labels)

            result_image[0 : yc, xc : 512, :] = image

            if boxes.shape[0] > 0:
                boxes[:, [0, 2]] += xc

            result_boxes.extend(boxes)

        elif i == 2:
            image, boxes = crop_image(image, boxes, labels, 0, 0, 512-xc, 512-yc)
            if len(boxes) > 0:
                result_labels.extend(labels)

            result_image[yc:512, xc:512, :] = image
            if boxes.shape[0] > 0:
                boxes[:, [0, 2]] += xc
                boxes[:, [1, 3]] += yc

            result_boxes.extend(boxes)

        else:
            image, boxes = crop_image(image, boxes, labels, 512-xc, 0, 512, 512-yc)
            if len(boxes) > 0:
                result_labels.extend(labels)

            result_image[yc : 512, 0 : xc, :] = image
            if boxes.shape[0] > 0:
                boxes[ :, [1, 3]] += yc

            result_boxes.extend(boxes)

    return result_image, result_boxes, result_labels


def mixup(data_list, bg_dir, min=0.4, max=0.5, alpha=1.0):
    image, bbox, labels = data_list[0]
    image = cv2.imread(image)

    bg_transform = A.Compose([
        A.Resize(512, 512, always_apply=True),

        A.OneOf([
            A.RandomRotate90(p=0.3),
            A.HorizontalFlip(p=0.3),
            A.VerticalFlip(p=0.3),
        ], p=1),

        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=(-0.25, 0.25), p=0.3),
            A.HueSaturationValue(val_shift_limit=(40, 80), p=0.3),
            A.ChannelShuffle(p=0.3)
        ], p=1)
    ])

    bg_file_list = glob(f"{bg_dir}/*")
    bg_index = random.randint(0, len(bg_file_list)-1)
    bg_image = cv2.imread(bg_file_list[bg_index])

    transformed = bg_transform(image=bg_image)
    background = transformed['image']

    lam = np.clip(np.random.beta(alpha, alpha), min, max)
    mixup_image = (lam*background + (1 - lam)*image).astype(np.uint8)

    return mixup_image, bbox, labels

    
def get_data(num, lines):
    data = []
    for n in range(num):
        index = np.random.randint(0, len(lines))
        line = lines[index].strip().split()

        image_path = line[0]
        labels = line[1:]

        bboxes, classes = [], []
        for label in labels:
            label = label.split(',')
            bbox = list(map(int, label[:4]))
            c = int(label[-1])

            bboxes.append(bbox)
            classes.append(c)

        data.append([image_path, bboxes, classes])

    return data


def main_func(txt_path):
    records = open(f"{SAVE_DIR}/list.txt", "w")
    
    lines = open(txt_path, "r").readlines()
    random.shuffle(lines)
    
    for steps in range(STEPS):
        for index in tqdm(range(len(lines))):
            opt = np.random.randint(0, 3)

            if opt == 0:
                data_list = get_data(4, lines)
                result_image, result_bboxes, result_classes = mosaic(data_list)

            elif opt == 1:
                data_list = get_data(1, lines)
                result_image, result_bboxes, result_classes = mixup(data_list, bg_dir=BG_DIR, min=0.2, max=0.3)

            else:
                data_list = get_data(1, lines)
                result_image, result_bboxes, result_classes = augmentation(data_list)

            file_name = f"{SAVE_DIR}/images/{index:>06}_{steps}.jpg"
            cv2.imwrite(f"{file_name}", result_image)
            records.write(f"{file_name}")

            for bbox, label in zip(result_bboxes, result_classes):
                records.write(' ')
                
                xmin, ymin, xmax, ymax = bbox
                c = label
                records.write(f"{int(xmin)},{int(ymin)},{int(xmax)},{int(ymax)},{int(c)}")

            records.write("\n")

            if VISUALIZE:
                sample = result_image.copy()
                for bbox in result_bboxes:
                    cv2.rectangle(sample, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color=(0, 0, 255))

                cv2.imshow("sample", sample)
                cv2.waitKey(0)

    records.close()


if __name__ == "__main__":
    DATA_DIR = "/data/Datasets/WIDER/CUSTOM"
    BG_DIR = "/data/Datasets/Mixup_background"
    SAVE_DIR = f"{DATA_DIR}/augmentation"
    VISUALIZE = True
    STEPS = 5

    train_txt = f"{DATA_DIR}/train/list.txt"

    transform = A.Compose([
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=(-0.25, 0.25), contrast_limit=(-0.3, 0.3), p=0.5),
            A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=(0, 0), val_shift_limit=(0, 3), p=0.5),
        ], p=1),

        A.OneOf([
            A.RandomRain(blur_value=4, brightness_coefficient=0.3, p=0.4),
            A.Downscale(scale_min=0.85, scale_max=0.95, p=0.3),
            A.MotionBlur(blur_limit=(3, 5), p=0.3)
        ], p=0.5),

        A.ShiftScaleRotate(p=0.5),

        A.OneOf([
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5)
        ], p=0.5)

    ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=['labels']))

    if not os.path.isdir(f"{SAVE_DIR}"):
        os.makedirs(f"{SAVE_DIR}/images")

    main_func(train_txt)