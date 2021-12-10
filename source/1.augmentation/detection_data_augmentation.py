import os
import cv2
import pathlib
import pandas as pd
import albumentations as A
import xml.etree.ElementTree as ET
from lxml.etree import Element, SubElement, tostring

def read_label(label_file: str):
    label_df = pd.read_csv(label_file, sep=',', index_col=False, header=None)
    labels = label_df[0].tolist()

    return labels


def read_xml(xml_file: str):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    objects = root.findall("object")
    
    bboxes, labels, areas = [], [], []
    if len(objects) > 0:
        class_names = [object.findtext("name") for object in objects]
        
        for idx, name in enumerate(class_names):
            if name in LABELS:
                bbox = objects[idx].find("bndbox")

                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)

                bboxes.append((xmin, ymin, xmax, ymax))
                labels.append(name)
                areas.append((xmax - xmin) * (ymax - ymin))

    return bboxes, labels, areas    


def get_files(dir: str):
    ds = pathlib.Path(dir)
    files = list(ds.glob('*'))
    files = [str(path) for path in files]

    # print(len(files))

    return files

def write_xml(bboxes, labels, filename, height, width, number):
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

    if len(bboxes) > 0:
        for idx in range(len(bboxes)):
            node_object = SubElement(node_root, "object")
            node_name = SubElement(node_object, "name")            
            node_name.text = str(labels[0])
            
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
            node_xmin.text = str(int(bboxes[idx][0]))
            node_ymin = SubElement(node_bndbox, 'ymin')
            node_ymin.text = str(int(bboxes[idx][1]))
            node_xmax = SubElement(node_bndbox, 'xmax')
            node_xmax.text = str(int(bboxes[idx][2]))
            node_ymax = SubElement(node_bndbox, 'ymax')
            node_ymax.text = str(int(bboxes[idx][3]))

    tree = ET.ElementTree(node_root)
    tree.write(f'{SAVE_DIR}/annotations/{filename}_{number}.xml')


def augmentation(image_path, bboxes, labels, areas):
    image = cv2.imread(image_path)
    filename = image_path.split('/')[-1].split('.')[0]

    for idx in range(AUG_N):
        transformed = transform(image=image, bboxes=bboxes, labels=labels)
        t_image = transformed['image']
        t_bboxes = transformed['bboxes']

        img_height, img_width = t_image.shape[:-1]
        cv2.imwrite(f"{SAVE_DIR}/images/{filename}_{idx}.jpg", t_image)
        write_xml(t_bboxes, labels, filename, img_height, img_width, idx)

        # for bbox, label in zip(t_bboxes, labels):
        #         cv2.rectangle(t_image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255))
        #         cv2.putText(t_image, label, (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

        # cv2.imshow('result', t_image)
        # cv2.waitKey(0)
        

def process(img_files: list, xml_files: list):
    for xml_file in xml_files:
        filename = xml_file.split('/')[-1].split('.')[0]

        if f"{IMAGES_DIR}/{filename}.jpg" in img_files:
            bboxes, labels, areas = read_xml(xml_file)
            print(bboxes, labels, areas)
            augmentation(f"{IMAGES_DIR}/{filename}.jpg", bboxes, labels, areas)

        else:
            print("Can't Find image file")


if __name__ == "__main__":
    ROOT_DIR = "/data/Datasets/SPC/Seeds/Foreground"
    LABEL_DIR = "/data/Datasets/SPC/Labels/labels.txt"
    SAVE_DIR = "/data/Datasets/SPC/set3/augmentations"
    AUG_N = 25

    transform = A.Compose([
        A.OneOf([
            A.Resize(448, 448),
            A.RandomSizedBBoxSafeCrop(448, 448),
        ], p=1),

        A.OneOf([
            A.Rotate(border_mode=0, limit=(-45, 45), p=1),
            A.ShiftScaleRotate(border_mode=0, rotate_limit=(-45, 45), p=1)
        ], p=1),

        A.OneOf([
            A.Cutout(num_holes=1, max_h_size=224, max_w_size=224, p=1),
            A.RGBShift(p=1),
            A.RandomBrightnessContrast(p=1)
        ], p=1),

        A.OneOf([
            A.ToGray(p=1),
            A.ToSepia(p=1)
        ], p=0.4)

    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], min_visibility=0.2))

    if not os.path.isdir(f"{SAVE_DIR}/images") and not os.path.isdir(f"{SAVE_DIR}/annotations"):
        os.makedirs(f"{SAVE_DIR}/images")
        os.makedirs(f"{SAVE_DIR}/annotations")

    
    IMAGES_DIR = f"{ROOT_DIR}/images"
    ANNOT_DIR = f"{ROOT_DIR}/annotations"
    LABELS = read_label(LABEL_DIR)
    # print(LABELS)

    img_files = get_files(IMAGES_DIR)
    xml_files = get_files(ANNOT_DIR)

    process(img_files, xml_files)