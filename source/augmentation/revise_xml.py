import os
import cv2
import pandas as pd
import albumentations as A
import xml.etree.ElementTree as ET
from tqdm import tqdm
from glob import glob
from lxml.etree import Element, SubElement, tostring
from src.utils import read_label_file, visualize, write_xml

def load_annot(xml_file):
    label_name = xml_file.split('/')[-4]
    tree = ET.parse(xml_file)
    root = tree.getroot()
    objects = root.findall("object")

    bboxes, labels = [], []
    if len(objects) > 0:
        class_names = [object.findtext("name") for object in objects]

        for idx, name in enumerate(class_names):
            if name != label_name:
                print(f"Wrong Label : {xml_file} ----> {name}")
                name = label_name

            bbox = objects[idx].find("bndbox")
            xmin = int(float(bbox.find('xmin').text))
            ymin = int(float(bbox.find('ymin').text))
            xmax = int(float(bbox.find('xmax').text))
            ymax = int(float(bbox.find('ymax').text))
                
            bboxes.append([xmin, ymin, xmax, ymax])
            labels.append(name)

    return bboxes, labels

if __name__ == "__main__":
    targets = ["Paris_baguette", "Dunkin", "Baskin_robbins"]
    root = "/data/Datasets/SPC"
    ds_path = f"{root}/Cvat"
    label_path = f"{root}/Labels/labels.txt"
    classes = read_label_file(label_path)
    print(classes)

    label_check = set()
    for target in targets:
        images = sorted(glob(f"{ds_path}/{target}/*/JPEGImages/*"))
        annotations = sorted(glob(f"{ds_path}/{target}/*/Annotations/*.xml"))
        dataset = list(zip(images, annotations))

        for i in tqdm(range(len(dataset))):
        # for i in range(len(dataset)):
            image = cv2.imread(images[i])
            height, width = image.shape[0], image.shape[1]

            bboxes, labels = load_annot(annotations[i])
            # visualize(image, bboxes, labels, format='pascal_voc')

            for label in labels:
                label_check.add(label)

            save_path = ('/').join((annotations[i].split('/')[:-2]))
            xml_file_name = annotations[i].split('/')[-1].split('.')[0]
            write_xml(save_path, bboxes, labels, xml_file_name, height, width, 'pascal_voc')

    print(label_check)
    if sorted(list(label_check)) == classes:
        print("No Issues")