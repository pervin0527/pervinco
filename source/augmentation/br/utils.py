import os
import cv2
import random
import numpy as np
import pandas as pd
import albumentations as A
import xml.etree.ElementTree as ET
from glob import glob
from lxml.etree import Element, SubElement


def read_label_file(label_file: str):
    label_df = pd.read_csv(label_file, sep=',', index_col=False, header=None)
    labels = label_df[0].tolist()

    return labels


def make_file_list(dir):
    image_files = sorted(glob(f"{dir}/*/JPEGImages/*"))
    annot_files = sorted(glob(f"{dir}/*/Annotations/*"))

    return image_files, annot_files


def get_content_filename(xml_file: str):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    filename = root.find("filename").text.split('.')[0]
    return filename


def convert_coordinates(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh

    return x, y, w, h


def read_xml(xml_file: str, classes: list, format):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    width = int(root.find('size').find('width').text)
    height = int(root.find('size').find('height').text)
    objects = root.findall("object")
    
    # bboxes, labels, areas = [], [], []
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

                if format == "yolo":
                    box = (float(xmin), float(xmax), float(ymin), float(ymax))
                    xmin, ymin, xmax, ymax = convert_coordinates((width, height), box)
                    name = classes.index(name)

                elif format == "albumentations":
                    xmin = int(xmin) / width
                    ymin = int(ymin) / height
                    xmax = int(xmax) / width
                    ymax = int(ymax) / height

                else:
                    xmin = int(xmin)
                    ymin = int(ymin)
                    xmax = int(xmax)
                    ymax = int(ymax)
                    
                bboxes.append([xmin, ymin, xmax, ymax])
                labels.append(name)
                # areas.append((xmax - xmin) * (ymax - ymin))

    # return bboxes, labels, areas    
    return bboxes, labels


def get_augmentation(transform):
    return A.Compose(transform, bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], min_area=0.5, min_visibility=0.2))


def visualize(image, boxes, labels, format="pascal_voc", show_info=True):
    if show_info:
        print(labels)
        print(boxes)
        print()
    
    for bb, c in zip(boxes, labels):       
        # print(c, bb)
        height, width = image.shape[:-1]

        if format == "pascal_voc":
            cv2.rectangle(image, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (255, 255, 0), thickness=1)
            cv2.putText(image, str(c), (int(bb[0]), int(bb[1])), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), thickness=1)            

        elif format == "albumentations":
            cv2.rectangle(image, (int(bb[0] * width + 0.5), int(bb[1] * height + 0.5)), (int(bb[2] * width + 0.5), int(bb[3] * height + 0.5)), (255, 255, 0), thickness=1)
            cv2.putText(image, str(c), (int(bb[0]), int(bb[1])), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), thickness=1)

    # image = cv2.resize(image, (960, 960))
    cv2.imshow('result', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def write_xml(save_path, bboxes, labels, filename, height, width, format):
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

            if format == "albumentations":
                xmin = int(xmin * width + 0.5)
                ymin = int(ymin * height + 0.5)
                xmax = int(xmax * width + 0.5)
                ymax = int(ymax * height + 0.5)

            elif format == "yolo":
                xmax = int((bbox[0]*width) + (bbox[2] * width)/2.0)
                xmin = int((bbox[0]*width) - (bbox[2] * width)/2.0)
                ymax = int((bbox[1]*height) + (bbox[3] * height)/2.0)
                ymin = int((bbox[1]*height) - (bbox[3] * height)/2.0)


            # print(xmin, ymin, xmax, ymax)
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
    
def make_save_dir(save_dir):
    try:
        if not os.path.isdir(f"{save_dir}/images") or os.path.isdir(f"{save_dir}/annotations"):
            os.makedirs(f"{save_dir}/images")
            os.makedirs(f"{save_dir}/annotations")
            os.makedirs(f"{save_dir}/result")

    except:
        pass

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
                    A.MotionBlur(p=0.3)
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