from genericpath import isfile
import os
import cv2
import albumentations as A
import xml.etree.ElementTree as ET
from collections import deque
from lxml.etree import Element, SubElement


def make_save_dir(dir):
    if not os.path.isdir(dir):
        os.makedirs(f"{dir}/images")
        os.makedirs(f"{dir}/annotations")
        os.makedirs(f"{dir}/img_with_bbox")


def make_label_field(n):
    labels = []
    for _ in range(n):
        labels.extend(['face'])

    return labels


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


def refine_coordinates(bbox, img_height, img_width):
    x, y, w, h = bbox
    xmin = x
    ymin = y
    xmax = w + xmin
    ymax = h + ymin

    outlier = None
    if xmin >= xmax or ymin >= ymax or xmin < 0 or ymin < 0 or xmax > img_width or ymax > img_height or (abs(xmax - xmin) * abs(ymax - ymin)) < MINIMUM_AREA:
        outlier = True

    return [xmin, ymin, xmax, ymax], outlier


def augmentation(image, bboxes, labels):
    transformed = transform(image=image, bboxes=bboxes, labels=labels)
    t_image, t_bboxes, t_labels = transformed['image'], transformed['bboxes'], transformed['labels']

    return t_image, t_bboxes, t_labels


def wider_data_process(txt, is_train):
    print(txt)
    if is_train:
        image_dir = f"{WIDER_DIR}/WIDER_train/images"
        save_dir = f"{WIDER_DIR}/CUSTOM/train_{IMG_SIZE}"
        make_save_dir(save_dir)

    else:
        image_dir = f"{WIDER_DIR}/WIDER_val/images"
        save_dir = f"{WIDER_DIR}/CUSTOM/test_{IMG_SIZE}"
        make_save_dir(save_dir)

    lines = open(txt, "r").readlines()
    lines = deque(lines)

    index = 0
    records = open(f"{save_dir}/list.txt", "w")
    while lines:
        image_file = lines.popleft()[:-1]
        num_boxes = int(lines.popleft()[:-1])

        if num_boxes > MAX_OBJECTS:
            for _ in range(num_boxes):
                lines.popleft()[:-1].split()

        elif num_boxes == 0:
            lines.popleft()[:-1].split()

        else:
            image = cv2.imread(f"{image_dir}/{image_file}")
            img_height, img_width = image.shape[:2]
            bboxes = []

            for _ in range(num_boxes):
                bbox_with_attr = lines.popleft()[:-1].split()
                bbox = list(map(int, bbox_with_attr[:4]))
                bbox, outlier = refine_coordinates(bbox, img_height, img_width)

                if outlier == None:
                    bboxes.append(bbox)

            if num_boxes == len(bboxes):
                labels = make_label_field(num_boxes)

                image, bboxes, labels = augmentation(image, bboxes, labels)

                img_with_bbox = image.copy()
                for bbox in bboxes:
                    cv2.rectangle(img_with_bbox, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0,0,255), 2)

                if VIS:
                    cv2.imshow("result", img_with_bbox)
                    cv2.waitKey(0)

                else:
                    cv2.imwrite(f"{save_dir}/images/{index:>07}.jpg", image)
                    write_xml(f"{save_dir}/annotations", bboxes, labels, f"{index:>07}", img_height, img_width)
                    records.write(f"{index:>07}\n")
                    cv2.imwrite(f"{save_dir}/img_with_bbox/{index:07}.jpg", img_with_bbox)
                    index += 1

        if not is_train and index == 100:
            break

    return records, index
            

if __name__ == "__main__":
    IMG_SIZE = 512
    CLASSES = ["face"]
    MAX_OBJECTS = 10
    MINIMUM_AREA = 5000
    VIS = False

    transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE, always_apply=True)
    ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=['labels']))

    WIDER_DIR = "/data/Datasets/WIDER"
    WIDER_TRAIN = f"{WIDER_DIR}/wider_face_split/wider_face_train_bbx_gt.txt"
    WIDER_TEST = f"{WIDER_DIR}/wider_face_split/wider_face_val_bbx_gt.txt"

    train_records, train_end_index = wider_data_process(WIDER_TRAIN, True)
    print(train_end_index)
    
    test_records, test_end_index = wider_data_process(WIDER_TEST, False)
    print(test_end_index)