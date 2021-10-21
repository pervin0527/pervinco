import os
import cv2
import pathlib
import xml.etree.ElementTree as ET

from xml.dom import minidom
from lxml.etree import Element, SubElement, tostring
from sklearn.model_selection import train_test_split


def get_file_list(path):
    ds = pathlib.Path(path)

    images = list(ds.glob('*.jpg'))
    images = [str(path) for path in images]

    return images


def read_annot_file(path):
    if os.path.isfile(path):
        tree = ET.parse(path)
        root = tree.getroot()
        objects = root.findall('object')

        annot_data = []
        if objects[0].find('bndbox') != None:
            for obj in objects:
                bboxes = obj.find('bndbox')
                names = obj.find('name')
                CLASSES.add(names.text)
                
                xmin = int(float(bboxes.find('xmin').text))
                ymin = int(float(bboxes.find('ymin').text))
                xmax = int(float(bboxes.find('xmax').text))
                ymax = int(float(bboxes.find('ymax').text))

                annot_data.append((names.text, xmin, ymin, xmax, ymax))

        return annot_data

    else:
        return None


def write_new_xml(org_data, index, width, height):
    node_root = Element("annotation")
    
    node_folder = SubElement(node_root, 'folder')
    node_folder.text = save_path.split('/')[-1]

    node_filename = SubElement(node_root, 'filename')
    node_filename.text = f'{index}.jpg'

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

    tree = ET.ElementTree(node_root)
    tree.write(f'{save_path}/annotations/{index}.xml')
    # print(f'{save_path}/annotations/{index}.xml')


if __name__ == "__main__":
    CLASSES = set()
    ds_path = "/data/Datasets/Seeds/VOCtrainval_11-May-2012/VOCdevkit/VOC2012"
    save_path = "/data/Datasets/Seeds/DMC/set8/"

    if not os.path.isdir(save_path):
        os.makedirs(f'{save_path}/images')
        os.makedirs(f'{save_path}/annotations')

    images_path = f"{ds_path}/JPEGImages"
    images_list = get_file_list(images_path)

    for idx, image_file in enumerate(images_list):
        image = cv2.imread(image_file)
        height, width = image.shape[:-1]

        filename = image_file.split('/')[-1].split('.')[0]
        xml_file = f"{ds_path}/Annotations/{filename}.xml"        
        xml_data = read_annot_file(xml_file)

        if "giant" not in xml_data:
            xml_data = []
            write_new_xml(xml_data, f"{idx}_voc", width, height)
            cv2.imwrite(f"{save_path}/images/{idx}_voc.jpg", image)

        else:
            write_new_xml(xml_data, f"{idx}_dmc", width, height)
            cv2.imwrite(f"{save_path}/images/{idx}_dmc.jpg", image)