import os
import cv2
import albumentations as A
import xml.etree.ElementTree as ET
from glob import glob
from tqdm import tqdm
from lxml.etree import Element, SubElement


def read_xml_file(xml_file):
    target = ET.parse(xml_file).getroot()

    height = int(target.find("size").find("height").text)
    width = int(target.find("size").find("width").text)

    bboxes, labels = [], []
    for obj in target.iter("object"):
        label = obj.find("name").text.strip()
        labels.append([label])

        bbox = obj.find("bndbox")
        pts = ["xmin", "ymin", "xmax", "ymax"]

        bnd_box = []
        for i, pt in enumerate(pts):
            current_pt = int(float(bbox.find(pt).text))
            if pt == "xmin" or pt == "ymin" and current_pt < 0:
                current_pt = 0
            elif pt == "xmax" and current_pt > width:
                current_pt = width
            elif pt == "ymax" and current_pt > height:
                current_pt = height
            bnd_box.append(current_pt)

        bboxes.append(bnd_box)

    return bboxes, labels

def get_files(dirs):
    total_images, total_annots = [], []
    for dir in dirs:
        image_files = sorted(glob(f"{dir}/*/JPEGImages/*"))
        annot_files = sorted(glob(f"{dir}/*/Annotations/*"))
        
        total_images.extend(image_files)
        total_annots.extend(annot_files)

    return list(zip(total_images, total_annots))


def write_xml(path, bboxes, labels):
    file_name = path.split('/')[-1].split('.')[0]
    
    root = Element("annotation")
    folder = SubElement(root, "folder")
    folder.text = "images"
    filename = SubElement(root, "filename")
    filename.text = f'{file_name}.jpg'
    
    size = SubElement(root, "size")
    w = SubElement(size, "width")
    w.text = str(img_width)
    h = SubElement(size, "height")
    h.text = str(img_height)
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
            xmin, ymin, xmax, ymax = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

            if xmin >= 0 and ymin >= 0 and xmax <= img_width and ymax <= img_height:
                node_xmin = SubElement(bndbox, 'xmin')
                node_xmin.text = str(int(xmin))
                node_ymin = SubElement(bndbox, 'ymin')
                node_ymin.text = str(int(ymin))
                node_xmax = SubElement(bndbox, 'xmax')
                node_xmax.text = str(int(xmax))
                node_ymax = SubElement(bndbox, 'ymax')
                node_ymax.text = str(int(ymax))
            
    tree = ET.ElementTree(root)
    tree.write(path)

    
def save_dataset(dataset):
    for index in tqdm(range(len(dataset))):
        img_file, xml_file = dataset[index]

        image = cv2.imread(img_file)
        bboxes, labels = read_xml_file(xml_file)

        resized = resize_transform(image=image, bboxes=bboxes, labels=labels)
        image, bboxes, labels = resized["image"], resized["bboxes"], resized["labels"]

        cv2.imwrite(f"{save_dir}/JPEGImages/{index:>06}.jpg", image)
        write_xml(f"{save_dir}/Annotations/{index:>06}.xml", bboxes, labels)


if __name__ == "__main__":
    data_dirs = ["/home/ubuntu/Datasets/SPC/Cvat/Baskin_robbins", "/home/ubuntu/Datasets/BR/cvat"]
    data_classes = ["Baskin_robbins"]
    save_dir = "/home/ubuntu/Datasets/BR/set1"
    
    img_height, img_width = (416, 416)
    resize_transform = A.Compose([
        A.Resize(height=img_height, width=img_width, always_apply=True)
    ], bbox_params=A.BboxParams(format='pascal_voc', min_area=0.5, min_visibility=0.2, label_fields=['labels']))

    if not os.path.isdir(save_dir):
        os.makedirs(f"{save_dir}/JPEGImages")
        os.makedirs(f"{save_dir}/Annotations")

    dataset = get_files(data_dirs)
    save_dataset(dataset)