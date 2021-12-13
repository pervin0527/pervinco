import os
import cv2
import pathlib
import pandas as pd
import albumentations as A
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


def get_content_filename(xml_file: str):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    filename = root.find("filename").text.split('.')[0]
    return filename


def convert_coordinates(width, height, xmin, ymin, xmax, ymax):
    x_center = (xmin + xmax) / (2 * width)
    y_center = (ymin + ymax) / (2 * height)
    width = (xmax - xmin) / width
    height = (ymax - ymin) / height

    return (x_center, y_center, width, height)


def read_xml(xml_file: str, classes: list, format):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    width = int(root.find('size').find('width').text)
    height = int(root.find('size').find('width').text)
    objects = root.findall("object")
    
    # bboxes, labels, areas = [], [], []
    bboxes, labels = [], []
    if len(objects) > 0:
        class_names = [object.findtext("name") for object in objects]
        
        for idx, name in enumerate(class_names):
            if name in classes:
                bbox = objects[idx].find("bndbox")

                xmin = int(float(bbox.find('xmin').text))
                ymin = int(float(bbox.find('ymin').text))
                xmax = int(float(bbox.find('xmax').text))
                ymax = int(float(bbox.find('ymax').text))

                # print("BEFORE : ", xmin, ymin, xmax, ymax)

                if format == "yolo":
                    # bboxes.append(convert_coordinates((width, height), (xmin, ymin, xmax, ymax)))
                    bboxes.append(convert_coordinates(width, height, xmin, ymin, xmax, ymax))
                    labels.append(classes.index(name))

                else:
                    bboxes.append([xmin, ymin, xmax, ymax])
                    labels.append(name)
                    # areas.append((xmax - xmin) * (ymax - ymin))

    # return bboxes, labels, areas    
    return bboxes, labels


def get_default_ds(image_dir: str, annot_dir: str, classes: list):
    image_list = get_files(image_dir)
    annot_list = get_files(annot_dir)

    ds = []
    for img_file, annot_file in zip(image_list, annot_list):
        if img_file.split('/')[-1].split('.')[0] == annot_file.split('/')[-1].split('.')[0] == get_content_filename(annot_file).split('.')[0]:
            bboxes, labels = read_xml(annot_file, classes)
            ds.append((img_file, bboxes, labels))

        else:
            print("Not matched files")
        
    return ds


def get_augmentation(transform):
    return A.Compose(transform, bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], min_area=0.5, min_visibility=0.2))


def visualize(image, boxes, labels, bbox_format="pascal_voc"):
    for bb, c in zip(boxes, labels):
        print(c, bb)
        height, width = image.shape[:-1]

        if bbox_format == "pascal_voc":
            cv2.rectangle(image, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (0, 255, 255))
            cv2.putText(image, str(c), (int(bb[0]), int(bb[1])), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0))            

        elif bbox_format == "albumentations":
            cv2.rectangle(image, (int(bb[0] * width + 0.5), int(bb[1] * height + 0.5)), (int(bb[2] * width + 0.5), int(bb[3] * height + 0.5)), (0, 255, 255))
            cv2.putText(image, str(c), (int(bb[0]), int(bb[1])), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0))

    image = cv2.resize(image, (1920, 1080))
    cv2.imshow('result', image)
    cv2.waitKey(0)


def write_xml(save_path, bboxes, labels, filename, height, width, format):
    root = Element("annotation")
    
    folder = SubElement(root, "folder")
    folder.text = "images"

    file_name = SubElement(root, "filename")
    file_name.text = f'{filename}.jpg'
    
    size = SubElement(root, "size")
    w = SubElement(size, "width")
    w.text = str(width)
    h = SubElement(size, "height")
    h.text = str(height)
    depth = SubElement(size, "depth")
    depth.text = "3"

    if labels:
        for label, bbox in zip(labels, bboxes):
            obj = SubElement(root, 'object')
            name = SubElement(obj, 'name')
            name.text = label
            pose = SubElement(obj, 'pose')
            pose.text = 'Unspecified'
            truncated = SubElement(obj, 'truncated')
            truncated.text = '0'
            difficult = SubElement(obj, 'difficult')
            difficult.text = '0'
            bndbox = SubElement(obj, 'bndbox')
            xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3]

            if format == "albumentations":
                xmin = int(xmin * width + 0.5)
                ymin = int(ymin * height + 0.5)
                xmax = int(xmax * width + 0.5)
                ymax = int(ymax * height + 0.5)

            elif format == "yolo":
                xmax = int((bbox[0]*width) + (bbox[2] * width)/2.0)
                xmin = int((bbox[0]*width) - (bbox[2] * width)/2.0)
                ymax = int((bbox[1]*height) + (bbox[3] * height)/2.0)
                ymin = int((bbox[1]*height) - (bbox[3] * height)/2.0)


            # print(xmin, ymin, xmax, ymax)
            node_xmin = SubElement(bndbox, 'xmin')
            node_xmin.text = str(xmin)
            node_ymin = SubElement(bndbox, 'ymin')
            node_ymin.text = str(ymin)
            node_xmax = SubElement(bndbox, 'xmax')
            node_xmax.text = str(xmax)
            node_ymax = SubElement(bndbox, 'ymax')
            node_ymax.text = str(ymax)
    
    tree = ET.ElementTree(root)    
    tree.write(f"{save_path}/{filename}.xml")
    
def make_save_dir(save_dir):
    try:
        if not os.path.isdir(f"{save_dir}/images") or os.path.isdir(f"{save_dir}/annotations"):
            os.makedirs(f"{save_dir}/images")
            os.makedirs(f"{save_dir}/annotations")

    except:
        pass