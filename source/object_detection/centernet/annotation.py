import os
import pandas as pd
import xml.etree.ElementTree as ET
from tqdm import tqdm
from glob import glob
from sklearn.model_selection import train_test_split

def read_label_file(txt_path):
    df = pd.read_csv(txt_path, sep=",", index_col=False, header=None)
    total_labels = df[0].to_list()
    print(total_labels)

    return total_labels

def split_dataset(data_dir):
    total_images = sorted(glob(f"{data_dir}/images/*"))
    total_xmls = sorted(glob(f"{data_dir}/annotations/*"))
    train_images, test_images, train_xmls, test_xmls = train_test_split(total_images, total_xmls, test_size=0.001, shuffle=False)

    print(len(train_images), len(train_xmls))
    print(len(test_images), len(test_xmls))

    return train_images, train_xmls, test_images, test_xmls

def write_annotation_txt(images, xmls, save_name):
    f = open(f"{data_dir}/{save_name}.txt", "w")
    for index in tqdm(range(len(images))):
        image_file = images[index]
        xml_file = xmls[index]

        annotation = image_file
        root = ET.parse(xml_file).getroot()        
        objects = root.findall("object")
        for obj in objects:
            bbox = obj.find("bndbox")
            class_index = classes.index(obj.find("name").text.lower().strip())
            xmin = bbox.find("xmin").text.strip()
            ymin = bbox.find("ymin").text.strip()
            xmax = bbox.find("xmax").text.strip()
            ymax = bbox.find("ymax").text.strip()
            annotation += " " + ",".join([xmin, ymin, xmax, ymax, str(class_index)])
        # print(annotation)
        f.write(annotation + "\n")
    
    print(f"{data_dir}/{save_name}.txt SAVED")


if __name__ == "__main__":
    data_dir = "/home/ubuntu/Datasets/COCO2017"
    label_txt = f"{data_dir}/Labels/labels.txt"
    classes = read_label_file(label_txt)

    train_images, train_xmls, test_images, test_xmls = split_dataset(data_dir)
    write_annotation_txt(train_images, train_xmls, "train")
    write_annotation_txt(test_images, test_xmls, "test")