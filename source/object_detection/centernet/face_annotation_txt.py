import os
import cv2
import numpy as np
import albumentations as A
from tqdm import tqdm
from collections import deque


def visualize(image, bboxes):
    vis_img = image.copy()

    for bbox in bboxes:
        cv2.rectangle(vis_img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color=(0, 0, 255), thickness=1)

    cv2.imshow("sample", vis_img)
    cv2.waitKey(0)


def augmentation(image, bboxes, labels):
    transformed = transform(image=image, bboxes=bboxes, labels=labels)
    t_image, t_bboxes, t_labels = transformed['image'], transformed['bboxes'], transformed['labels']

    return t_image, t_bboxes, t_labels


def refine_coordinates(bbox):
    x, y, w, h = bbox
    xmin = x
    ymin = y
    xmax = w + xmin
    ymax = h + ymin

    return [xmin, ymin, xmax, ymax]


def make_label_field(n):
    labels = []
    for _ in range(n):
        labels.extend(['face'])

    return labels


def write_annots(augment_bboxes, augment_labels, f):
    for (bbox, label) in zip(augment_bboxes, augment_labels):
        f.write(' ')
        xmin, ymin, xmax, ymax = bbox
        c = LABELS.index(label)
        f.write(f"{int(xmin)},{int(ymin)},{int(xmax)},{int(ymax)},{int(c)}")
    f.write("\n")


def read_txt(txt_path, image_dir, save_dir):
    if not os.path.isdir(save_dir):
        os.makedirs(f"{save_dir}/images")

    record = open(f"{save_dir}/list.txt", "w")
    f = open(txt_path, "r")
    lines = f.readlines()
    lines = deque(lines)

    idx = 0
    while lines:
        img_path = lines.popleft()[:-1] ## except '\n'
        n_boxes = int(lines.popleft()[:-1])

        if 0 < n_boxes <= MAX_OBJECTS:
            print(img_path, n_boxes)

            bboxes = []
            for _ in range(int(n_boxes)):
                bbox_with_attr = lines.popleft()[:-1].split()
                bbox = list(map(int, bbox_with_attr[:4]))
                bbox = refine_coordinates(bbox)
                bboxes.append(bbox)

            image = cv2.imread(f"{image_dir}/images/{img_path}")
            classes = make_label_field(int(n_boxes))

            try:
                augment_image, augment_bboxes, augment_classes = augmentation(image, bboxes, classes)
                file_name = f"{idx:>06}.jpg"
                cv2.imwrite(f"{save_dir}/images/{file_name}", augment_image)
                
                record.write(f"{save_dir}/images/{file_name}")
                write_annots(augment_bboxes, augment_classes, record)
                idx += 1

            except:
                pass
                
        elif n_boxes > MAX_OBJECTS:
            for _ in range(n_boxes):
                lines.popleft()[:-1].split()

        elif n_boxes == 0:
            lines.popleft()[:-1].split()

        
if __name__ == "__main__":
    ROOT_DIR = "/home/ubuntu/Datasets/WIDER"
    ANNOT_DIR = [f"{ROOT_DIR}/wider_face_split/wider_face_train_bbx_gt.txt",
                 f"{ROOT_DIR}/wider_face_split/wider_face_val_bbx_gt.txt"]
    LABELS = ["face"]
    MAX_OBJECTS = 3

    transform = A.Compose([
        A.Resize(512, 512, always_apply=True)
    ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=['labels']))

    for annot_file in ANNOT_DIR:
        if annot_file.split('/')[-1] == "wider_face_train_bbx_gt.txt":
            print("wider_face_train_bbx_gt")
            IMG_DIR = f"{ROOT_DIR}/WIDER_train"
            SAVE_DIR = f"{ROOT_DIR}/CUSTOM/train"

        else:
            print("wider_face_test_bbx_gt")
            IMG_DIR = f"{ROOT_DIR}/WIDER_val"
            SAVE_DIR = f"{ROOT_DIR}/CUSTOM/test"
        
        read_txt(annot_file, IMG_DIR, SAVE_DIR)
