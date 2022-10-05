import pandas as pd
import xml.etree.ElementTree as ET
from glob import glob
from tqdm import tqdm
from data_utils import read_label_file

def read_xml_file(path):
    tree = ET.parse(path)
    root = tree.getroot()
    objects = root.findall("object")
    
    bboxes, labels = [], []
    if len(objects) > 0:
        class_names = [object.findtext("name") for object in objects]
        
        for idx, name in enumerate(class_names):
            if name in class_list:
                bbox = objects[idx].find("bndbox")

                xmin = int(float(bbox.find('xmin').text))
                ymin = int(float(bbox.find('ymin').text))
                xmax = int(float(bbox.find('xmax').text))
                ymax = int(float(bbox.find('ymax').text))               
                    
                bboxes.append([xmin, ymin, xmax, ymax])
                labels.append(name)

    return bboxes, labels


def convert_annotation(images, xmls, mode="train"):
    f = open(f"{data_path}/{mode}.txt", "w", encoding="utf-8")
    for index in tqdm(range(len(xmls))):
        image_file, xml_file = images[index], xmls[index]

        f.write(f"{image_file}")
        bboxes, labels = read_xml_file(xml_file)

        for bbox, label in zip(bboxes, labels):
            f.write(" ")
            xmin, ymin, xmax, ymax = bbox
            class_id = class_list.index(label)
            f.write(f"{xmin},{ymin},{xmax},{ymax},{class_id}")
        f.write("\n")
    f.close()


if __name__ == "__main__":
    data_path = "/data/Datasets/VOCdevkit/VOC2012/detection"
    label_path = "/data/Datasets/VOCdevkit/VOC2012/detection/Labels/labels.txt"
    folder_name = ["train", "valid"]

    class_list = read_label_file(label_path)
    print(class_list)

    train_images, train_xmls = sorted(glob(f"{data_path}/{folder_name[0]}/images/*.jpg")), sorted(glob(f"{data_path}/{folder_name[0]}/annotations/*.xml"))
    valid_images, valid_xmls = sorted(glob(f"{data_path}/{folder_name[1]}/images/*.jpg")), sorted(glob(f"{data_path}/{folder_name[1]}/annotations/*.xml"))
    print(len(train_images), len(train_xmls))
    print(len(valid_images), len(valid_xmls))

    convert_annotation(train_images, train_xmls, "train")
    convert_annotation(valid_images, valid_xmls, "valid")