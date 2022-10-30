import os
import cv2
import math
import random
import numpy as np
import albumentations as A
import xml.etree.ElementTree as ET
from glob import glob
from tqdm import tqdm
from copy import deepcopy

def make_file_list(data_list):
    image_files, xml_files = [], []
    for path in data_list:
        images = sorted(glob(f"{path}/*/JPEGImages/*"))
        xmls = sorted(glob(f"{path}/*/Annotations/*"))
        print(len(images), len(xmls))

        image_files.extend(images)
        xml_files.extend(xmls)

    return image_files, xml_files


def read_xml_file(xml_file):
    target = ET.parse(xml_file).getroot()

    bboxes, labels = [], []
    for obj in target.iter("object"):
        label = obj.find("name").text.strip()
        labels.append([label])

        bbox = obj.find("bndbox")
        pts = ["xmin", "ymin", "xmax", "ymax"]

        bnd_box = []
        for pt in pts:
            current_pt = int(float(bbox.find(pt).text))
            bnd_box.append(current_pt)
        bboxes.append(bnd_box)

    return bboxes, labels


def get_aug_params(value, center=0):
    if isinstance(value, float):
        return random.uniform(center - value, center + value)
    elif len(value) == 2:
        return random.uniform(value[0], value[1])
    else:
        raise ValueError("Affine params should be either a sequence containing two values\
                          or single float values. Got {}".format(value))


def get_affine_matrix(target_size, degrees=10, translate=0.1, scales=0.1, shear=10,):
    twidth, theight = target_size

    angle = get_aug_params(degrees)
    scale = get_aug_params(scales, center=1.0)

    if scale <= 0.0:
        raise ValueError("Argument scale should be positive")

    R = cv2.getRotationMatrix2D(angle=angle, center=(0, 0), scale=scale)
    M = np.ones([2, 3])

    shear_x = math.tan(get_aug_params(shear) * math.pi / 180)
    shear_y = math.tan(get_aug_params(shear) * math.pi / 180)

    M[0] = R[0] + shear_y * R[1]
    M[1] = R[1] + shear_x * R[0]

    translation_x = get_aug_params(translate) * twidth  # x translation (pixels)
    translation_y = get_aug_params(translate) * theight  # y translation (pixels)

    M[0, 2] = translation_x
    M[1, 2] = translation_y

    return M, scale


def apply_affine_to_bboxes(targets, target_size, M, scale):
    num_gts = len(targets)

    twidth, theight = target_size
    corner_points = np.ones((4 * num_gts, 3))
    corner_points[:, :2] = targets[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(4 * num_gts, 2)
    corner_points = corner_points @ M.T
    corner_points = corner_points.reshape(num_gts, 8)

    corner_xs = corner_points[:, 0::2]
    corner_ys = corner_points[:, 1::2]
    new_bboxes = (np.concatenate((corner_xs.min(1), corner_ys.min(1), corner_xs.max(1), corner_ys.max(1))).reshape(4, num_gts).T)

    new_bboxes[:, 0::2] = new_bboxes[:, 0::2].clip(0, twidth)
    new_bboxes[:, 1::2] = new_bboxes[:, 1::2].clip(0, theight)

    targets[:, :4] = new_bboxes

    return targets


def random_affine(img, targets=(), target_size=(640, 640), degrees=10, translate=0.1, scales=0.1, shear=10):
    M, scale = get_affine_matrix(target_size, degrees, translate, scales, shear)
    img = cv2.warpAffine(img, M, dsize=target_size, borderValue=(114, 114, 114))

    if len(targets) > 0:
        targets = apply_affine_to_bboxes(targets, target_size, M, scale)

    return img, targets


def get_mosaic_coordinate(mosaic_image, mosaic_index, xc, yc, w, h, input_h, input_w):
    if mosaic_index == 0:
        x1, y1, x2, y2 = max(xc - w, 0), max(yc - h, 0), xc, yc
        small_coord = w - (x2 - x1), h - (y2 - y1), w, h

    elif mosaic_index == 1:
        x1, y1, x2, y2 = xc, max(yc - h, 0), min(xc + w, input_w * 2), yc
        small_coord = 0, h - (y2 - y1), min(w, x2 - x1), h

    elif mosaic_index == 2:
        x1, y1, x2, y2 = max(xc - w, 0), yc, xc, min(input_h * 2, yc + h)
        small_coord = w - (x2 - x1), 0, w, min(y2 - y1, h)

    elif mosaic_index == 3:
        x1, y1, x2, y2 = xc, yc, min(xc + w, input_w * 2), min(input_h * 2, yc + h)
        small_coord = 0, 0, min(w, x2 - x1), min(y2 - y1, h)

    return (x1, y1, x2, y2), small_coord


def apply_augmentation(dataset):
    datasets = deepcopy(dataset)

    for n in tqdm(range(n_output)):
        if len(datasets) == 1:
            datasets = deepcopy(dataset)

        idx = random.randint(0, len(datasets))
        data = datasets[idx]
        del datasets[idx]      
        
        if apply_mosaic and random.random() < mosaic_prob:
            mosaic_labels = []
            yc = int(random.uniform(0.5 * img_size, 1.5 * img_size))
            xc = int(random.uniform(0.5 * img_size, 1.5 * img_size))
            indices = [idx] + [random.randint(0, len(datasets) - 1) for _ in range(3)]
            
            for i_mosaic, index in enumerate(indices):
                img_file, annot_file = datasets[index]
                bboxes, classes = read_xml_file(annot_file)
                img = cv2.imread(img_file)
                h0, w0 = img.shape[:2]
                scale = min(1. * img_size / h0, 1. * img_size / w0)

                img = cv2.resize(img, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_LINEAR)
                (h, w, c) = img.shape[:3]

                if i_mosaic == 0:
                    mosaic_img = np.full((img_size * 2, img_size * 2, c), 114, dtype=np.uint8)

                (l_x1, l_y1, l_x2, l_y2), (s_x1, s_y1, s_x2, s_y2) = get_mosaic_coordinate(mosaic_img, i_mosaic, xc, yc, w, h, img_size, img_size)
                mosaic_img[l_y1:l_y2, l_x1:l_x2] = img[s_y1:s_y2, s_x1:s_x2]
                padw, padh = l_x1 - s_x1, l_y1 - s_y1

                labels = np.empty((0, 5))
                for bbox, label in zip(bboxes, classes):
                    data = [bbox[0], bbox[1], bbox[2], bbox[3], data_labels.index(label[0])]
                    labels = np.vstack((labels, data))

                if labels.size > 0:
                    labels[:, 0] = scale * labels[:, 0] + padw
                    labels[:, 1] = scale * labels[:, 1] + padh
                    labels[:, 2] = scale * labels[:, 2] + padw
                    labels[:, 3] = scale * labels[:, 3] + padh
                mosaic_labels.append(labels)

            if len(mosaic_labels):
                mosaic_labels = np.concatenate(mosaic_labels, 0)
                np.clip(mosaic_labels[:, 0], 0, 2 * img_size, out=mosaic_labels[:, 0])
                np.clip(mosaic_labels[:, 1], 0, 2 * img_size, out=mosaic_labels[:, 1])
                np.clip(mosaic_labels[:, 2], 0, 2 * img_size, out=mosaic_labels[:, 2])
                np.clip(mosaic_labels[:, 3], 0, 2 * img_size, out=mosaic_labels[:, 3])

            mosaic_img, mosaic_labels = random_affine(mosaic_img,
                                                      mosaic_labels,
                                                      target_size=(img_size, img_size),
                                                      degrees=10.0,
                                                      translate=0.1,
                                                      scales=(0.1, 1.5),
                                                      shear=2.0)

            for mosaic_label in mosaic_labels:
                xmin, ymin, xmax, ymax, _ = int(mosaic_label[0]), int(mosaic_label[1]), int(mosaic_label[2]), int(mosaic_label[3]), int(mosaic_label[4])
                cv2.rectangle(mosaic_img, (xmin, ymin), (xmax, ymax), color=(0, 0, 255), thickness=2)
            cv2.imshow("result", mosaic_img)
            cv2.waitKey(0)
                    

        # else:
        #     image_file, annot_file = data[0], data[1]
        #     print(image_file, annot_file)
        #     ## normal single image augmentation
        #     image = cv2.imread(image_file)
        #     bboxes, labels = read_xml_file(annot_file)

        #     transfromed = basic_transform(image=image, bboxes=bboxes, labels=labels)
        #     image_t, bboxes_t = transfromed["image"], transfromed["bboxes"]
        
if __name__=="__main__":
    data_list = ["/data/Datasets/SPC/Cvat/Baskin_robbins", "/data/Datasets/BR/cvat"]
    data_labels = ["Baskin_robbins"]
    img_size = 640
    n_output = 10
    
    apply_mosaic = True
    mosaic_prob = 1.0

    basic_transform = A.Compose([
            A.Resize(img_size, img_size, p=1),
            A.RandomBrightnessContrast(p=1, brightness_limit=(-0.2, 0.2)),
    ], bbox_params=A.BboxParams(format='pascal_voc', min_area=0.5, min_visibility=0.2, label_fields=['labels']))

    image_files, xml_files = make_file_list(data_list)
    dataset = list(zip(image_files, xml_files))
    apply_augmentation(dataset)

    print(len(dataset))