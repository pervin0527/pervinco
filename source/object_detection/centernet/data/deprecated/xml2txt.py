import os
import cv2
import pathlib
import pandas as pd
from glob import glob
import xml.etree.ElementTree as ET
from lxml.etree import Element, SubElement, tostring

def read_label_file(label_file: str):
    label_df = pd.read_csv(label_file, sep=',', index_col=False, header=None)
    labels = label_df[0].tolist()

    return labels


def get_files(dir: str):
    ds = pathlib.Path(dir)
    files = list(ds.glob('*'))
    files = sorted([str(path) for path in files])

    return files


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

def yolo2voc(class_id, width, height, x, y, w, h):
    xmin = int((x*width) - (w * width)/2.0)
    ymin = int((y*height) - (h * height)/2.0)
    xmax = int((x*width) + (w * width)/2.0)
    ymax = int((y*height) + (h * height)/2.0)
    class_id = int(class_id)

    return (class_id, xmin, ymin, xmax, ymax)


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

if __name__ == "__main__":
    FOLDERS = ["train", "test"]
    ROOT_DIR = "/data/Datasets/WIDER/CUSTOM_XML"
    LABEL_DIR = "/data/Datasets/WIDER/Labels/labels.txt"

    classes = read_label_file(LABEL_DIR)
    print(classes)

    for folder in FOLDERS:
        images = get_files(f"{ROOT_DIR}/{folder}/images")
        annotations = get_files(f"{ROOT_DIR}/{folder}/annotations")

        if not os.path.isdir(f"{ROOT_DIR}/{folder}/v4set"):
            os.makedirs(f"{ROOT_DIR}/{folder}/v4set")

        print(f"{ROOT_DIR}/{folder}")
        for annot in annotations:
            file_name = annot.split('/')[-1].split('.')[0]
            image = cv2.imread(f"{ROOT_DIR}/{folder}/images/{file_name}.jpg")
            cv2.imwrite(f"{ROOT_DIR}/{folder}/v4set/{file_name}.jpg", image)

            with open(f"{ROOT_DIR}/{folder}/v4set/{file_name}.txt", "w") as f:
                bboxes, labels = read_xml(annot, classes, format="yolo")
                for bbox, label in zip(bboxes, labels):
                    print(str(int(label)) + " " + " ".join([("%.6f" % a) for a in bbox]) + '\n')
                    f.write(str(int(label)) + " " + " ".join([("%.6f" % a) for a in bbox]) + '\n')
                f.close()

        v4_images = glob(f"{ROOT_DIR}/{folder}/v4set/*.jpg")
        with open(f"{ROOT_DIR}/{folder}/files.txt", "w") as f:
            for image in v4_images:
                data = image + '\n'
                f.write(data)
        f.close()