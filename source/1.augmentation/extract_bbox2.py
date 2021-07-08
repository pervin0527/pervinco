import os
import sys
import cv2
import pathlib
import random
import xml.etree.ElementTree as ET

def show_sample():
    sample_xml = "/home/barcelona/labelImg/fire_extinguisher_115.xml"
    sample_img = "/home/barcelona/labelImg/fire_extinguisher_115.jpg"

    annot_data = get_bboxes(sample_xml)

    for idx in range(len(annot_data)):
        label_name = annot_data[idx][0]
        xmin, ymin, xmax, ymax = annot_data[idx][1], annot_data[idx][2], annot_data[idx][3], annot_data[idx][4]

        print(xmin, ymin, xmax, ymax)

        if not os.path.isdir(f"/data/Datasets/Seeds/test/{label_name}"):
            os.makedirs(f"/data/Datasets/Seeds/test/{label_name}")

        img = cv2.imread(sample_img, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        height, width, _ = img.shape

        if ymax > height:
            ymax = height

        if xmax > width:
            xmax = width

        crop_img = img[ymax:ymin, xmin:xmax]
        crop_img = cv2.resize(crop_img, (224, 224))
        cv2.imshow('result', crop_img)
        cv2.waitKey(0)


def get_bboxes(label_path):
    tree = ET.parse(label_path)
    root = tree.getroot()
    obj_xml = root.findall('object')
    
    try:
        if obj_xml[0].find('bndbox') != None:
            result = []

            for obj in obj_xml:
                bbox_original = obj.find('bndbox')
                names = obj.find('name')
            
                xmin = int(float(bbox_original.find('xmin').text))
                ymin = int(float(bbox_original.find('ymin').text))
                xmax = int(float(bbox_original.find('xmax').text))
                ymax = int(float(bbox_original.find('ymax').text))

                result.append((names.text, xmin, ymin, xmax, ymax))

            return result

    except:
        return None
        

if __name__ == "__main__":
    img_path = "/data/Datasets/Seeds/ETRI_detection/images"
    annot_path = "/data/Datasets/Seeds/ETRI_detection/annotations"
    
    images = pathlib.Path(img_path)
    annots = pathlib.Path(annot_path)

    images = list(images.glob('*.jpg'))
    annots = list(annots.glob('*.xml'))
    
    images = sorted([str(path) for path in images])
    annots = sorted([str(path) for path in annots])

    error_list = []
    for image, annot in zip(images, annots):
        
        img_file_name = image.split('/')[-1]
        xml_file_name = annot.split('/')[-1]

        img_file_name = img_file_name.split('.')[0]
        xml_file_name = xml_file_name.split('.')[0]

        if img_file_name == xml_file_name:
            annot_data = get_bboxes(annot)

            if annot_data != None:
                for idx in range(len(annot_data)):
                    label_name = annot_data[idx][0]
                    xmin, ymin, xmax, ymax = annot_data[idx][1], annot_data[idx][2], annot_data[idx][3], annot_data[idx][4]

                    img = cv2.imread(image)
                    height, width, _ = img.shape                        

                    if not os.path.isdir(f"/data/Datasets/Seeds/test/{label_name}"):
                        os.makedirs(f"/data/Datasets/Seeds/test/{label_name}")

                    try:
                        crop_img = img[ymin:ymax, xmin:xmax]
                        crop_img = cv2.resize(crop_img, (224, 224))
                        cv2.imwrite(f"/data/Datasets/Seeds/test/{label_name}/{img_file_name}.jpg", crop_img)

                    except:
                        error_list.append(img_file_name)

        else:
            print(img_file_name, xml_file_name)

    print(len(error_list))

    for e in error_list:
        os.system(f'sudo rm {img_path}/{e}.jpg')
        os.system(f'sudo rm {annot_path}/{e}.xml')

    print("done")