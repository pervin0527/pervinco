import cv2
import xml.etree.ElementTree as ET
from glob import glob

def read_xml(xml_file, classes):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    width = int(root.find('size').find('width').text)
    height = int(root.find('size').find('height').text)
    objects = root.findall("object")
    
    bboxes, labels = [], []
    if len(objects) > 0:
        class_names = [object.findtext("name") for object in objects]
        
        for idx, name in enumerate(class_names):
            if name in classes:
                bbox = objects[idx].find("bndbox")

                xmin = float(bbox.find('xmin').text)
                ymin = float(bbox.find('ymin').text)
                xmax = float(bbox.find('xmax').text)
                ymax = float(bbox.find('ymax').text)               

                xmin = int(xmin)
                ymin = int(ymin)
                xmax = int(xmax)
                ymax = int(ymax)
                    
                bboxes.append([xmin, ymin, xmax, ymax])
                labels.append(name)

    return bboxes, labels


if __name__ == "__main__":
    classes = ["face"]
    image_dir = "/data/Datasets/FACE_DETECTION/augmentation/images"
    annotation_dir = "/data/Datasets/FACE_DETECTION/augmentation/annotations"

    image_files = sorted(glob(f"{image_dir}/*.jpg"))
    xml_files = sorted(glob(f"{annotation_dir}/*.xml"))

    for image, xml in zip(image_files, xml_files):
        image = cv2.imread(image)
        bboxes, labels = read_xml(xml, classes)

        sample = image.copy()
        for bbox in bboxes:
            xmin, ymin, xmax, ymax = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            cv2.rectangle(sample, (xmin, ymin), (xmax, ymax), color=(0, 0, 255))
        cv2.imshow("sample", sample)
        cv2.waitKey(0)