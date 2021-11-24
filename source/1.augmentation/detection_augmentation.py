import os
import cv2
import pathlib
import numpy as np
import pandas as pd
import albumentations as A
import xml.etree.ElementTree as ET
from lxml.etree import Element, SubElement, tostring

def visualize(image, boxes, labels):
    for bb, c in zip(boxes, labels):
        # print(c, bb)
        cv2.rectangle(image, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (0, 255, 255))
        cv2.putText(image, str(c), (int(bb[0]), int(bb[1])), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0))

    cv2.imshow('result', image)
    cv2.waitKey(0)    


def read_label(label_path):
    label_file = pd.read_csv(label_path, sep=',', index_col=False, header=None)
    label_map = label_file[0].tolist()

    return label_map


def get_file_list(data_path):
    ds = pathlib.Path(data_path)

    image_list = list(ds.glob('images/*.jpg'))
    images = [str(path) for path in image_list]

    return images


def read_xml(xml_path):
    annotation = ET.parse(xml_path)
    objects = annotation.findall("object")
    
    is_target = False
    annot_data = []
    if objects:
        names = [obj.findtext("name") for obj in objects]

        for idx, name in enumerate(names):
            if name in CLASSES:
                bbox = objects[idx].find('bndbox')

                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)
                
                annot_data.append([name, xmin, ymin, xmax, ymax])

    if len(annot_data) > 0:
        is_target = True

    return annot_data, is_target


def write_xml(annot_data, filename, height, width, count):
    node_root = Element("annotation")
    
    node_folder = SubElement(node_root, "folder")
    node_folder.text = "images"

    node_filename = SubElement(node_root, "filename")
    node_filename.text = f'{filename}.jpg'
    
    node_size = SubElement(node_root, "size")
    node_width = SubElement(node_size, "width")
    node_width.text = str(width)
    node_height = SubElement(node_size, "height")
    node_height.text = str(height)
    node_depth = SubElement(node_size, "depth")
    node_depth.text = "3"

    if len(annot_data) > 0:
        for idx in range(len(annot_data)):
            # print(annot_data, annot_data[1])
            node_object = SubElement(node_root, "object")
            node_name = SubElement(node_object, "name")
            node_name.text = str(annot_data[idx][0])

            after.add(str(annot_data[idx][0]))
            
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
            node_xmin.text = str(int(annot_data[idx][1]))
            node_ymin = SubElement(node_bndbox, 'ymin')
            node_ymin.text = str(int(annot_data[idx][2]))
            node_xmax = SubElement(node_bndbox, 'xmax')
            node_xmax.text = str(int(annot_data[idx][3]))
            node_ymax = SubElement(node_bndbox, 'ymax')
            node_ymax.text = str(int(annot_data[idx][4]))

    tree = ET.ElementTree(node_root)
    tree.write(f'{output_path}/annotations/{filename}_{count}.xml')


def data_augment(image, annot):
    bboxes = []
    labels = []
    for data in annot:
        labels.append([data[0]])
        bboxes.append([data[1], data[2], data[3], data[4]])

    transform = A.Compose([
        A.Rotate(p=1, border_mode=1),
    
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

    transformed = transform(image=image, bboxes=bboxes, labels=labels)
    visualize(transformed['image'], transformed['bboxes'], transformed['labels'])

    return transformed['image'], transformed['bboxes'], transformed['labels']


def data_processing(image_files):
    for image in image_files:
        filename = image.split('/')[-1].split('.')[0]

        image = cv2.imread(image)
        img_height, img_width = image.shape[:-1]

        xml_file = f"{data_path}/annotations/{filename}.xml"
        if os.path.isfile(xml_file):
            data, target = read_xml(xml_file)

            if target:
                for idx in range(repeat):
                    transformed_image, transformed_box, transformed_label = data_augment(image, data)
                    transformed_data = []
                    for label, bbox in zip(transformed_label, transformed_box):
                        transformed_data.append([label[0], bbox[0], bbox[1], bbox[2], bbox[3]])

                    # print(transformed_data)
                    write_xml(transformed_data, filename, img_height, img_width, idx+1)
                    cv2.imwrite(f'{output_path}/images/{filename}_{idx+1}.jpg', transformed_image)

            else:
                write_xml(data, filename, img_height, img_width, 0)
                cv2.imwrite(f'{output_path}/images/{filename}_0.jpg', image)
      
            
if __name__ == "__main__":
    label_file = "/data/Datasets/Seeds/SPC/Labels/labels.txt"
    data_path = "./data"
    output_path = "./data/output"
    repeat = 5
    
    if not os.path.isdir(f"{output_path}/annotations") or not os.path.isdir(f"{output_path}/images"):
        os.makedirs(f"{output_path}/images")
        os.makedirs(f"{output_path}/annotations")

    after = set()
    CLASSES = read_label(label_file)
    print("LABELS : ", CLASSES)
    
    images = get_file_list(data_path)
    print(len(images))

    data_processing(images)
    print(after)