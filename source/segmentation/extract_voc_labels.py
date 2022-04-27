import os
from glob import glob
import xml.etree.ElementTree as ET
from lxml.etree import Element, SubElement, tostring

def read_xml(xml_file):
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

if __name__ == "__main__":
    CLASSES = set()
    DATA_DIR = "/data/Datasets/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/Annotations"
    annotations = sorted(glob(f"{DATA_DIR}/*.xml"))

    for annot in annotations:
        bboxes, labels =read_xml(annot)
        
        for label in labels:
            CLASSES.add(label)

    for label in sorted(list(CLASSES)):
        print(label)