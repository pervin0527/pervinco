import os
import cv2
import random
import numpy as np
import albumentations as A
import xml.etree.ElementTree as ET

from glob import glob
from tqdm import tqdm
from lxml.etree import Element, SubElement


def write_xml(save_path, bboxes, labels, filename, height, width):
    root = Element("annotation")
    
    folder = SubElement(root, "folder")
    folder.text = "images"

    file_name = SubElement(root, "filename")
    file_name.text = f'{filename}.jpg'
    
    size = SubElement(root, "size")
    w = SubElement(size, "width")
    w.text = str(width)
    h = SubElement(size, "height")
    h.text = str(height)
    depth = SubElement(size, "depth")
    depth.text = "3"

    if labels:
        for label, bbox in zip(labels, bboxes):
            obj = SubElement(root, 'object')
            name = SubElement(obj, 'name')
            name.text = label
            pose = SubElement(obj, 'pose')
            pose.text = 'Unspecified'
            truncated = SubElement(obj, 'truncated')
            truncated.text = '0'
            difficult = SubElement(obj, 'difficult')
            difficult.text = '0'
            bndbox = SubElement(obj, 'bndbox')
            xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3]

            node_xmin = SubElement(bndbox, 'xmin')
            node_xmin.text = str(int(xmin))
            node_ymin = SubElement(bndbox, 'ymin')
            node_ymin.text = str(int(ymin))
            node_xmax = SubElement(bndbox, 'xmax')
            node_xmax.text = str(int(xmax))
            node_ymax = SubElement(bndbox, 'ymax')
            node_ymax.text = str(int(ymax))
    
    tree = ET.ElementTree(root)    
    tree.write(f"{save_path}/{filename}.xml")


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
            A.RandomBrightnessContrast(brightness_limit=(-0.15, 0.25), p=0.3),
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


def read_xml(xml_file, classes):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    width = int(root.find('size').find('width').text)
    height = int(root.find('size').find('height').text)
    objects = root.findall("object")
    
    bboxes, labels = [], []
    if len(objects) > 0:
        class_names = [object.findtext("name") for object in objects]
        
        for idx, name in enumerate(class_names):
            if name in classes:
                bbox = objects[idx].find("bndbox")

                xmin = float(bbox.find('xmin').text)
                ymin = float(bbox.find('ymin').text)
                xmax = float(bbox.find('xmax').text)
                ymax = float(bbox.find('ymax').text)               

                xmin = int(xmin)
                ymin = int(ymin)
                xmax = int(xmax)
                ymax = int(ymax)
                    
                bboxes.append([xmin, ymin, xmax, ymax])
                labels.append(name)

    return bboxes, labels


def load_file(files):
    data_list = []
    for file in files:
        image_path = f"{IMG_PATH}/{file}.jpg"
        xml_path = f"{ANNOT_PATH}/{file}.xml"
        
        bboxes, labels = read_xml(xml_path, CLASSES)
        data_list.append([image_path, bboxes, labels])

    return data_list

    
def read_file_list(path):
    total_files = open(path, "r").readlines()
    total_files = [file.strip() for file in total_files]
    random.shuffle(total_files)

    if not os.path.isdir(f"{SAVE_DIR}"):
        os.makedirs(f"{SAVE_DIR}/images")
        os.makedirs(f"{SAVE_DIR}/annotations")
    record = open(f"{SAVE_DIR}/list.txt", "w")

    for step in range(STEPS):
        for index in tqdm(range(len(total_files))):
            opt = random.randint(0, 2)
            if opt == 0:
                files = random.sample(total_files, 4)
                data_list = load_file(files)
                result_image, result_bboxes, result_classes = mosaic(data_list)

            elif opt == 1:
                files = random.sample(total_files, 1)
                data_list = load_file(files)
                result_image, result_bboxes, result_classes = mixup(data_list, bg_dir=BG_DIR, min=0.1, max=0.25)

            else:
                files = random.sample(total_files, 1)
                data_list = load_file(files)
                result_image, result_bboxes, result_classes = augmentation(data_list)

            if VISUALIZE:
                sample = result_image.copy()
                for bbox in result_bboxes:
                    cv2.rectangle(sample, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color=(0, 0, 255))

                cv2.imshow("sample", sample)
                cv2.waitKey(0)

            cv2.imwrite(f"{SAVE_DIR}/images/{step}_{index:>06}.jpg", result_image)
            write_xml(f"{SAVE_DIR}/annotations", result_bboxes, result_classes, f"{step}_{index:>06}", result_image.shape[0], result_image.shape[1])
            record.write(f"{step}_{index:>06}\n")


if __name__ == "__main__":
    DATA_DIR = "/data/Datasets/WFLW/CUSTOM_XML"
    SAVE_DIR = f"{DATA_DIR}/augmentation"
    IMG_PATH = f"{DATA_DIR}/train/images"
    ANNOT_PATH = f"{DATA_DIR}/train/annotations"
    TXT_PATH = f"{DATA_DIR}/train/list.txt"

    CLASSES = ["face"]
    VISUALIZE = False
    STEPS = 3
    BG_DIR = "/data/Datasets/Mixup_background"

    transform = A.Compose([
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=(-0.15, 0.25), contrast_limit=(-0.15, 0.25), p=0.5),
            A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=(0, 0), val_shift_limit=(0, 3), p=0.5),
        ], p=1),

        A.OneOf([
            A.RandomRain(blur_value=4, brightness_coefficient=0.3, p=0.4),
            A.Downscale(scale_min=0.85, scale_max=0.95, p=0.3),
            A.MotionBlur(blur_limit=(3, 5), p=0.3)
        ], p=0.5),

        A.ShiftScaleRotate(shift_limit=(-0.15, 0.15), scale_limit=(-0.15, 0.15), rotate_limit=(0, 0), border_mode=0, p=0.5),

        A.OneOf([
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5)
        ], p=0.5)

    ], bbox_params=A.BboxParams(format="pascal_voc", min_area=0.5, min_visibility=0.2, label_fields=['labels']))

    read_file_list(TXT_PATH)