import os
import cv2
import albumentations as A
import xml.etree.ElementTree as ET
from collections import deque
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


def augmentation(image, bboxes, labels):
    transformed = transform(image=image, bboxes=bboxes, labels=labels)
    t_image, t_bboxes, t_labels = transformed['image'], transformed['bboxes'], transformed['labels']

    return t_image, t_bboxes, t_labels


def make_label_field(n):
    labels = []
    for _ in range(n):
        labels.extend(['face'])

    return labels


def refine_coordinates(bbox):
    x, y, w, h = bbox
    xmin = x
    ymin = y
    xmax = w + xmin
    ymax = h + ymin

    return [xmin, ymin, xmax, ymax]


def read_txt(txt_path, image_dir, save_dir):
    if not os.path.isdir(save_dir):
        os.makedirs(f"{save_dir}/images")
        os.makedirs(f"{save_dir}/annotations")

    record = open(f"{save_dir}/list.txt", "w")
    f = open(txt_path, "r")
    lines = f.readlines()
    lines = deque(lines)

    idx = 0
    print(save_dir)
    while lines:
        img_path = lines.popleft()[:-1] ## except '\n'
        n_boxes = int(lines.popleft()[:-1])

        if 0 < n_boxes <= MAX_OBJECTS:
            # print(img_path, n_boxes)

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

                cv2.imwrite(f"{save_dir}/images/{idx:>06}.jpg", augment_image)
                write_xml(f"{save_dir}/annotations", augment_bboxes, augment_classes, f"{idx:>06}", augment_image.shape[0], augment_image.shape[1])
                record.write(f"{idx:>06}\n")
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
    MAX_OBJECTS = 10

    transform = A.Compose([
        A.Resize(384, 384, always_apply=True)
    ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=['labels']))


    for annot_file in ANNOT_DIR:
        if annot_file.split('/')[-1] == "wider_face_train_bbx_gt.txt":
            IMG_DIR = f"{ROOT_DIR}/WIDER_train"
            SAVE_DIR = f"{ROOT_DIR}/CUSTOM_XML/train"

        else:
            IMG_DIR = f"{ROOT_DIR}/WIDER_val"
            SAVE_DIR = f"{ROOT_DIR}/CUSTOM_XML/test"
        
        read_txt(annot_file, IMG_DIR, SAVE_DIR)