import os
import cv2
import random
import pathlib
import pandas as pd
import xml.etree.ElementTree as ET

from shutil import copyfile
from xml.dom import minidom
from lxml.etree import Element, SubElement, tostring


def choice_bg(image_list, annot_list):
        rand_idx = random.sample(range(len(image_list)), int(len(main_images) * bg_ratio))
        # print(len(rand_idx))
        
        images = []
        annots = []
        for idx in rand_idx:
            images.append(image_list[idx])
            annots.append(annot_list[idx])

        return images, annots


def get_file_list(path, is_main):
    ds_path = pathlib.Path(path)

    images = sorted(list(ds_path.glob('images/*.jpg')))
    annotations = sorted(list(ds_path.glob('annotations/*.xml')))

    images = [str(path) for path in images]
    annotations = [str(path) for path in annotations]

    if is_main == False:
        images, annotations = choice_bg(images, annotations)
        print(f"BG IMAGES : {len(images)}")
        print(f"BG ANNOTATIONS : {len(annotations)}")

    else:
        print(f"MAIN IMAGES : {len(images)}")
        print(f"MAIN ANNOTATIONS : {len(annotations)}")

    return images, annotations


def read_annot_file(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objects = root.findall('object')

    annot_data = []
    if len(objects) != 0:
        for obj in objects:
            bboxes = obj.find('bndbox')
            names = obj.find('name')
            
            xmin = int(float(bboxes.find('xmin').text))
            ymin = int(float(bboxes.find('ymin').text))
            xmax = int(float(bboxes.find('xmax').text))
            ymax = int(float(bboxes.find('ymax').text))

            annot_data.append((names.text, xmin, ymin, xmax, ymax))

    return annot_data


def write_new_xml(org_data, img_filename, width, height):
    node_root = Element("annotation")
    
    node_folder = SubElement(node_root, 'folder')
    node_folder.text = "images"

    node_filename = SubElement(node_root, 'filename')
    node_filename.text = f'{img_filename}'

    node_size = SubElement(node_root, 'size')

    node_width = SubElement(node_size, 'width')
    node_width.text = str(width)

    node_height = SubElement(node_size, 'height')
    node_height.text = str(height)

    node_depth = SubElement(node_size, 'depth')
    node_depth.text = '3'

    if org_data != None:
        for i in range(len(org_data)):
            node_object = SubElement(node_root, 'object')
            node_name = SubElement(node_object,'name')
            node_name.text = str(org_data[i][0])
            check_list.add(str(org_data[i][0]))

            node_pose = SubElement(node_object, 'pose')
            node_pose.text = 'Unspecified'

            node_truncated = SubElement(node_object, 'truncated')
            node_truncated.text = '0'

            node_occluded = SubElement(node_object, 'occluded')
            node_occluded.text = '0'

            node_difficult = SubElement(node_object, 'difficult')
            node_difficult.text = '0'

            node_bndbox = SubElement(node_object, 'bndbox')
            node_xmin = SubElement(node_bndbox, 'xmin')
            node_xmin.text = str(org_data[i][1])
            node_ymin = SubElement(node_bndbox, 'ymin')
            node_ymin.text = str(org_data[i][2])
            node_xmax = SubElement(node_bndbox, 'xmax')
            node_xmax.text = str(org_data[i][3])
            node_ymax = SubElement(node_bndbox, 'ymax')
            node_ymax.text = str(org_data[i][4])

    xml_filename = img_filename.split('.')[0]
    tree = ET.ElementTree(node_root)
    tree.write(f'{output_path}/annotations/{xml_filename}.xml')

def main_process(images, annotations):
    for img_file, xml_file in zip(images, annotations):
        img_filename = img_file.split('/')[-1]
        image = cv2.imread(img_file)
        img_height, img_width = image.shape[:-1]
        cv2.imwrite(f'{output_path}/images/{img_filename}', image)

        annot_data = read_annot_file(xml_file)
        annot_data = [label for label in annot_data if label[0] in CLASSES]
        # print(annot_data)
        write_new_xml(annot_data, img_filename, img_width, img_height)


if __name__ == "__main__":
    dataset_path = "/data/Datasets/Seeds/DMC/Samples2"
    label_path = "/data/Datasets/Seeds/DMC/Labels/labels.txt"
    bgset_path = "/data/Datasets/Seeds/COCO2017"
    output_path = "/data/Datasets/Seeds/DMC/test"
    bg_ratio = 0.2

    CLASSES = pd.read_csv(label_path, sep=' ', index_col=False, header=None)
    CLASSES = CLASSES[0].tolist()
    print(CLASSES)

    check_list = set()

    if not os.path.isdir(f'{output_path}/images'):
        os.makedirs(f'{output_path}/images')

    if not os.path.isdir(f'{output_path}/annotations'):
        os.makedirs(f'{output_path}/annotations')

    main_images, main_annots = get_file_list(dataset_path, True)
    bg_images, bg_annots = get_file_list(bgset_path, False)

    main_process(main_images, main_annots)
    main_process(bg_images, bg_annots)

    print(check_list)