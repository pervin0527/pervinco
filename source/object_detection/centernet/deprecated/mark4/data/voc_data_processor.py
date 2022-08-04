import os
import xml.etree.ElementTree as ET

def convert_voc_annotation(data_path, txt_file, annot_path, use_difficul_bbox=True):
    data_list_file = f"{data_path}/ImageSets/Main/{txt_file}.txt"
    with open(data_list_file, "r") as f:
        lines = f.readlines()
        files = [line.strip() for line in lines]

    with open(annot_path, 'a') as f:
        for file in files:
            img_path = f"{data_path}/JPEGImages/{file}.jpg"
            xml_path = f"{data_path}/Annotations/{file}.xml"

            annotation = img_path

            root = ET.parse(xml_path).getroot()
            objects = root.findall("object")
            for obj in objects:
                difficult = obj.find("difficult").text.strip()

                if (not use_difficul_bbox) and (int(difficult) == 1):
                    continue

                bbox = obj.find("bndbox")
                class_ind = classes.index(obj.find("name").text.lower().strip())
                xmin = bbox.find('xmin').text.strip()
                xmax = bbox.find('xmax').text.strip()
                ymin = bbox.find('ymin').text.strip()
                ymax = bbox.find('ymax').text.strip()
                annotation += ' ' + ','.join([xmin, ymin, xmax, ymax, str(class_ind)])

            print(annotation)
            f.write(annotation + "\n")

    return len(file)


if __name__ == "__main__":
    # classes = ["face"]
    classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
               'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
               'train', 'tvmonitor']

    data_path = "/data/Datasets/VOCdevkit/VOC2012"
    train_annotation = "./voc_train.txt"
    test_annotation = "./voc_test.txt"

    dataset1 = convert_voc_annotation(data_path, "train", train_annotation, False)