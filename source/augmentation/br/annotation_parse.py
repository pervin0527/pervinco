import os
import cv2
import xml.etree.ElementTree as ET
from glob import glob

def get_annot_data(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    bboxes, classes = [], []
    for obj in root.iter("object"):
        class_str = obj.find("name").text
        bounding_box = obj.find("bndbox")
        box = (int(float(bounding_box.find('xmin').text)), int(float(bounding_box.find('ymin').text)), int(float(bounding_box.find('xmax').text)), int(float(bounding_box.find('ymax').text)))

        classes.append(class_str)
        bboxes.append(box)

    return bboxes, classes

def draw_annot_data(annot_path):
    folders = sorted(glob(f"{annot_path}/*"))
    # folders = ["/data/Datasets/BR/cvat/FLOW압구정현대_12"]
    
    for folder in folders:
        xml_files = sorted(glob(f"{folder}/Annotations/*.xml"))
        spot_name = folder.split('/')[-1]

        if not os.path.isdir(f"{folder}/JPEGImages"):
            os.makedirs(f"{folder}/JPEGImages")

        for xml_file in xml_files:
            file_name = xml_file.split('/')[-1].split('.')[0]
            
            image_path = f"{images_dir}/{spot_name}/{file_name}.jpg"
            print(image_path)
            image = cv2.imread(image_path)

            result = image.copy()
            bboxes, classes = get_annot_data(xml_file)
            for bbox in bboxes:
                print(bbox)
                xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3]
                cv2.rectangle(result, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), thickness=3)

            # result = cv2.resize(result, (960, 720))
            # cv2.imshow(f"frame", result)
            # cv2.waitKey(0)

            cv2.imwrite(f"{folder}/JPEGImages/{file_name}.jpg", image)

if __name__ == "__main__":
    data_root = "/data/Datasets/BR"
    images_dir = f"{data_root}/frames"
    annotations_dir = f"{data_root}/cvat"
    draw_annot_data(annotations_dir)