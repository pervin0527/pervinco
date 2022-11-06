import os
import random
import xml.etree.ElementTree as ET
from lxml.etree import Element, SubElement

from tqdm import tqdm
from glob import glob

def make_file_list(dirs, num_valid=0):
    total_files = []
    for dir in dirs:
        images = sorted(glob(f"{dir}/JPEGImages/*"))
        annotations = sorted(glob(f"{dir}/Annotations/*"))
        print(f"{dir} - image_files : {len(images)},  annot_files : {len(annotations)}")

        total_files.extend(list(zip(images, annotations)))

    print(f"total_files : {len(total_files)}")

    if num_valid > 0:
        random.shuffle(total_files)
        train_files = total_files[:-num_valid]
        valid_files = total_files[-num_valid:]
        print(f"Train files : {len(train_files)}")
        print(f"Valid files : {len(valid_files)}")

        return train_files, valid_files

    else:
        return total_files

def make_save_dir(dir):
    if not os.path.isdir(dir):
        os.makedirs(f"{dir}/JPEGImages")
        os.makedirs(f"{dir}/Annotations")
        os.makedirs(f"{dir}/Results")

def load_annot_data(annot_file, target_classes):
    target = ET.parse(annot_file).getroot()

    height = int(target.find('size').find('height').text)
    width = int(target.find('size').find('width').text)

    bboxes, labels = [], []
    for obj in target.iter("object"):
        label = obj.find("name").text.strip()
        if label in target_classes:
            labels.append([label])

            bndbox = obj.find("bndbox")
            bbox = []
            for current in ["xmin", "ymin", "xmax", "ymax"]:
                coordinate = int(float(bndbox.find(current).text))
                if current == "xmin" and coordinate < 0:
                    coordinate = 0
                elif current == "ymin" and coordinate < 0:
                    coordinate = 0
                elif current == "xmax" and coordinate > width:
                    coordinate = width
                elif current == "ymax" and coordinate > height:
                    coordinate = height
                bbox.append(coordinate)
            bboxes.append(bbox)

    return bboxes, labels

def annot_write(dst, bboxes, labels, img_size):
    root = Element("annotation")
    folder = SubElement(root, "folder")
    folder.text = "JPEGImages"
    filename = SubElement(root, "filename")
    filename.text = dst.split('/')[-1].split('.')[0] + ".jpg"

    size = SubElement(root, "size")
    h = SubElement(size, "height")
    h.text = str(img_size[0])
    w = SubElement(size, "width")
    w.text = str(img_size[1])
    depth = SubElement(size, "depth")
    depth.text = "3"

    if labels:
        for label, bbox in zip(labels, bboxes):
            obj = SubElement(root, 'object')
            name = SubElement(obj, 'name')
            name.text = label[0]
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
    tree.write(dst)

if __name__ == "__main__":
    annotation_path = "/home/ubuntu/Datasets/BR/set3_384"
    folders = ["train", "valid"]
    classes = ["Baskin_robbins"] # Baskin_robbins

    check = 0
    for folder in folders:
        xml_files = sorted(glob(f"{annotation_path}/{folder}/Annotations/*.xml"))
        for idx in tqdm(range(len(xml_files))):
            xml_file = xml_files[idx]
            bboxes, labels = load_annot_data(xml_file, classes)

            for label in labels:
                if not label[0] in classes:
                    print(label)

    #         for label in labels:
    #             if not label in classes:
    #                 filename = xml_file.split('/')[-1].split('.')[0]
    #                 os.remove(f"{annotation_path}/{folder}/JPEGImages/{filename}.jpg")
    #                 os.remove(xml_file)
    #                 check += 1
    #         # annot_write(xml_file, bboxes, labels, (384, 384))

    # xml_files = sorted(glob(f"{annotation_path}/{folder}/Annotations/*.xml"))
    # print(len(xml_files))