import cv2
import os
import xml.etree.ElementTree as ET
from glob import glob
from xml.dom import minidom
from lxml.etree import Element, SubElement, tostring

def write_dummy(output_path, img_filename, width, height):
    node_root = Element("annotation")
    
    node_folder = SubElement(node_root, 'folder')
    node_folder.text = "images"

    node_filename = SubElement(node_root, 'filename')
    node_filename.text = f'{img_filename}.jpg'

    node_size = SubElement(node_root, 'size')

    node_width = SubElement(node_size, 'width')
    node_width.text = str(width)

    node_height = SubElement(node_size, 'height')
    node_height.text = str(height)

    node_depth = SubElement(node_size, 'depth')
    node_depth.text = '3'

    xml_filename = img_filename.split('.')[0]
    tree = ET.ElementTree(node_root)
    tree.write(f'{output_path}/annotations/{xml_filename}.xml')

if __name__ == "__main__":
    dummy_roots = "/data/Datasets/Seeds/SPC/Background"
    dummy_images = sorted(glob(f"{dummy_roots}/images/*.jpg"))
    print(len(dummy_images))

    for file in dummy_images:
        print(file)
        filename = file.split('/')[-1].split('.')[0]
        output_path = '/'.join(file.split('/')[:-2])
        # print(output_path)

        image = cv2.imread(file)
        height, width, _ = image.shape

        write_dummy(output_path, filename, width, height)
        # break