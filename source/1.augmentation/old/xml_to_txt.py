import os
import sys
import glob
import pandas as pd
import xml.etree.ElementTree as ET
from xml.dom import minidom
from src.utils import read_label_file

# def convert_coordinates(width, height, xmin, ymin, xmax, ymax):
#     x_center = (xmin + xmax) / (2 * width)
#     y_center = (ymin + ymax) / (2 * height)
#     width = (xmax - xmin) / width
#     height = (ymax - ymin) / height

#     return (x_center, y_center, width, height)


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

    return (x,y,w,h)


def convert_xml2yolo(lut, input_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for fname in glob.glob(input_path + "/*.xml"):
        filename = fname.split('/')[-1]
        filename = filename.split('.')[0]

        tree = ET.parse(fname)
        root = tree.getroot()

        obj_xml = root.findall("object")
        size_xml = root.find('size')

        image_width = int(size_xml.find("width").text)
        image_hegiht = int(size_xml.find("height").text)

        try:
            with open(output_path + '/' + filename + ".txt", 'w') as f:
                for obj in obj_xml:
                    class_id = obj.find("name").text
                    if class_id in lut:
                        label_str = str(lut.index(class_id))
                    else:
                        label_str = "-1"
                        print ("warning: label '%s' not in look-up table" %class_id)

                    bboxes = obj.find("bndbox")
                    xmin = float(float(bboxes.find('xmin').text))
                    ymin = float(float(bboxes.find('ymin').text))
                    xmax = float(float(bboxes.find('xmax').text))
                    ymax = float(float(bboxes.find('ymax').text))

                    box = (float(xmin), float(xmax), float(ymin), float(ymax))
                    # print(filename, image_width, image_hegiht, class_id, xmin, ymin, xmax, ymax)
                    result = convert_coordinates((image_width, image_hegiht), box)
                    # print(filename)

                    f.write(label_str + " " + " ".join([("%.6f" % a) for a in result]) + '\n')
        except:
            print(filename)


if __name__ == '__main__':
    input_path = "/data/Datasets/SPC/set1/train/augmetations/annotations"
    label_path = "/data/Datasets/SPC/set1/Labels/labels.txt"
    output_path = "/data/Datasets/SPC/set1/train/augmentations/labels"
    txt_filename = "train.txt"

    classes = read_label_file(label_path)
    convert_xml2yolo(classes, input_path, output_path)

    txt_path = f"{'/'.join(input_path.split('/')[:-1])}/{txt_filename}.txt"
    file = open(txt_path, 'w')

    for name in glob.iglob(os.path.join(f"{'/'.join(input_path.split('/')[:-1])}/images", "*.jpg")):  
        title, ext = os.path.splitext(os.path.basename(name))
        file.write(f"{'/'.join(input_path.split('/')[:-1])}/images" + "/" + title + '.jpg' + "\n")