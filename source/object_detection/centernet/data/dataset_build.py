import os
import cv2
import random
import albumentations as A
import xml.etree.ElementTree as ET
from tqdm import tqdm
from glob import glob
from collections import deque
from lxml.etree import Element, SubElement


def make_save_dir(dir):
    if not os.path.isdir(dir):
        os.makedirs(f"{dir}/images")
        os.makedirs(f"{dir}/annotations")
        os.makedirs(f"{dir}/img_with_bbox")


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


def convert_coordinates(height, width, x, y, w, h):
    xmax = int((x*width) + (w * width)/2.0)
    xmin = int((x*width) - (w * width)/2.0)
    ymax = int((y*height) + (h * height)/2.0)
    ymin = int((y*height) - (h * height)/2.0)
    
    return [xmin, ymin, xmax, ymax]



def augmentation(image, bboxes, labels):
    transformed = transform(image=image, bboxes=bboxes, labels=labels)
    t_image, t_bboxes, t_labels = transformed['image'], transformed['bboxes'], transformed['labels']

    return t_image, t_bboxes, t_labels


def wider_data_process(txt, is_train):
    print(txt)
    if is_train:
        image_dir = f"{WIDER_DIR}/WIDER_train/images"
        save_dir = f"{WIDER_DIR}/{FOLDER}/train_{IMG_SIZE}"
        make_save_dir(save_dir)

    else:
        image_dir = f"{WIDER_DIR}/WIDER_val/images"
        save_dir = f"{WIDER_DIR}/{FOLDER}/test_{IMG_SIZE}"
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
            labels = []

            for _ in range(num_boxes):
                bbox_with_attr = lines.popleft()[:-1].split()
                bbox = list(map(int, bbox_with_attr[:4]))
                attributes = list(map(int, bbox_with_attr[4:]))

                bbox, outlier = refine_coordinates(bbox, img_height, img_width)

                if outlier == None:
                    bboxes.append(bbox)
                    labels.extend([CLASSES[0]])

                    # if attributes[3] == 1:
                    #     labels.extend([CLASSES[1]])
                    # else:
                    #     labels.extend([CLASSES[0]])

            if num_boxes == len(bboxes):
                image, bboxes, labels = augmentation(image, bboxes, labels)

                img_with_bbox = image.copy()
                for bbox in bboxes:
                    cv2.rectangle(img_with_bbox, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0,0,255), 2)

                cv2.imwrite(f"{save_dir}/images/wider_{index}.jpg", image)
                write_xml(f"{save_dir}/annotations", bboxes, labels, f"wider_{index}", img_height, img_width)
                records.write(f"wider_{index}\n")
                cv2.imwrite(f"{save_dir}/img_with_bbox/wider_{index}.jpg", img_with_bbox)
                index += 1

        if not is_train and index == 50:
            break

    return index


def vw_data_process():
    image_dir = f"{VW_DIR}/images"
    annot_dir = f"{VW_DIR}/labels"

    annotations = sorted(glob(f"{annot_dir}/*.txt"))
    random.shuffle(annotations)
    split = [annotations[:num_trainset], annotations[-100:]]
    save_dirs = [f"{WIDER_DIR}/{FOLDER}/train_{IMG_SIZE}", f"{WIDER_DIR}/{FOLDER}/test_{IMG_SIZE}"]
    
    for i, dataset in enumerate(split):
        save_dir = save_dirs[i]
        make_save_dir(save_dir)
        records = open(f"{save_dir}/list.txt", "a")

        for index in tqdm(range(len(dataset))):
            file = dataset[index]
            file_name = file.split('/')[-1].split('.')[0]
            image_file = f"{image_dir}/{file_name}.jpg"

            image = cv2.imread(image_file)
            data = open(file, 'r')
            lines = data.readlines()
            labels, bboxes = [], []
            for line in lines:
                line = line.strip().split()
                label = int(line[0])
                x, y, w, h = list(map(float, line[1:]))
                xmin, ymin, xmax, ymax = convert_coordinates(image.shape[0], image.shape[1], x, y, w, h)
                labels.append(CLASSES[label])
                bboxes.append([xmin, ymin, xmax, ymax])
                
                image, bboxes, labels = augmentation(image, bboxes, labels)

                img_with_bbox = image.copy()
                for bbox in bboxes:
                    cv2.rectangle(img_with_bbox, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0,0,255), 2)

            cv2.imwrite(f"{save_dir}/images/vw_{index}.jpg", image)
            write_xml(f"{save_dir}/annotations", bboxes, labels, f"vw_{index}", image.shape[0], image.shape[1])
            records.write(f"vw_{index}\n")
            cv2.imwrite(f"{save_dir}/img_with_bbox/vw_{index}.jpg", img_with_bbox)


if __name__ == "__main__":
    IMG_SIZE = 512
    FOLDER = "FACE"
    CLASSES = ["face"]
    MAX_OBJECTS = 10
    num_trainset = 11200
    MINIMUM_AREA = 5000

    transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE, always_apply=True)
    ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=['labels']))

    WIDER_DIR = "/data/Datasets/WIDER"
    # WIDER_TRAIN = f"{WIDER_DIR}/wider_face_split/wider_face_train_bbx_gt.txt"
    # WIDER_TEST = f"{WIDER_DIR}/wider_face_split/wider_face_val_bbx_gt.txt"

    # train_end_index = wider_data_process(WIDER_TRAIN, True)
    # test_end_index = wider_data_process(WIDER_TEST, False)
    # print(train_end_index, test_end_index)

    VW_DIR = "/data/Datasets/300VW_Dataset_2015_12_14/total"
    vw_data_process()