import os
import cv2
import random
import albumentations as A
import xml.etree.ElementTree as ET

from glob import glob
from tqdm import tqdm
from lxml.etree import Element, SubElement


def write_xml(save_path, bboxes, labels, filename, height, width, format):
    root = Element("annotation")
    
    folder = SubElement(root, "folder")
    folder.text = "JPEGImages"

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
            node_xmin.text = str(int(xmin))
            node_ymin = SubElement(bndbox, 'ymin')
            node_ymin.text = str(int(ymin))
            node_xmax = SubElement(bndbox, 'xmax')
            node_xmax.text = str(int(xmax))
            node_ymax = SubElement(bndbox, 'ymax')
            node_ymax.text = str(int(ymax))
    
    tree = ET.ElementTree(root)    
    tree.write(f"{save_path}/{filename}.xml")


def visualize(image, bboxes):
    vis_img = image.copy()
    for bbox in bboxes:
        xmin, ymin, xmax, ymax = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        cv2.rectangle(vis_img, (xmin, ymin), (xmax, ymax), color=(0, 0, 255), thickness=3)

    return vis_img


def augmentation(image, bboxes, labels):
    transformed = transform(image=image, bboxes=bboxes, labels=labels)
    t_image, t_bboxes, t_labels = transformed["image"], transformed["bboxes"], transformed["labels"]

    return t_image, t_bboxes, t_labels


def refine_coordinates(height, width, x, y, w, h):
    xmax = int((x*width) + (w * width)/2.0)
    xmin = int((x*width) - (w * width)/2.0)
    ymax = int((y*height) + (h * height)/2.0)
    ymin = int((y*height) - (h * height)/2.0)
    
    return [xmin, ymin, xmax, ymax]


def record_xml_process(save_dir, txt_files):
    print(len(txt_files))

    for index in tqdm(range(len(txt_files))):
        file = txt_files[index]
        file_name = file.split('/')[-1].split('.')[0]
        image_file = f"{save_dir}/Frames/{file_name}.jpg"
        
        if os.path.isfile(image_file):
            image = cv2.imread(image_file)
            
            data = open(file, "r")
            lines = data.readlines()
            labels, bboxes = [], []
            for line in lines:
                line = line.strip().split()
                
                label = int(line[0])
                x, y, w, h = list(map(float, line[1:]))
                xmin, ymin, xmax, ymax = refine_coordinates(image.shape[0], image.shape[1], x, y, w, h)
                labels.append(classes[label])
                bboxes.append([xmin, ymin, xmax, ymax])

            # result_image, result_bboxes, result_labels = augmentation(image, bboxes, labels)
            # write_xml(f"{save_dir}/Annotations", result_bboxes, result_labels, f"{index:>09}", image.shape[0], image.shape[1], format="pascal_voc")
            # cv2.imwrite(f"{save_dir}/JPEGImages/{index:>09}.jpg", result_image)
            
            # result = visualize(result_image, result_bboxes)
            # cv2.imwrite(f"{save_dir}/Results/{index:>09}.jpg", result)

            write_xml(f"{save_dir}/Annotations", bboxes, labels, f"{index:>09}", image.shape[0], image.shape[1], format="pascal_voc")
            cv2.imwrite(f"{save_dir}/JPEGImages/{index:>09}.jpg", image)
            
            result = visualize(image, bboxes)
            cv2.imwrite(f"{save_dir}/Results/{index:>09}.jpg", result)


if __name__ == "__main__":
    path = "/home/ubuntu/Datasets/BR/total"
    image_size = 384
    classes = ["Baskin_robbins"]

    transform = A.Compose([
        A.Resize(image_size, image_size, p=1)
    ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]))

    txt_files = sorted(glob(f"{path}/exp/labels/*.txt"))

    if not os.path.isdir(f"{path}/JPEGImages") and not os.path.isdir(f"{path}/Annotations"):
        os.makedirs(f"{path}/JPEGImages")
        os.makedirs(f"{path}/Annotations")
        os.makedirs(f"{path}/Results")

    random.shuffle(txt_files)
    record_xml_process(path, txt_files)