import random
import cv2
import os
import time
import sys
import glob
import datetime
from matplotlib import pyplot as plt
import xml.etree.ElementTree as ET
from xml.dom import minidom


def get_boxes(label_path):
    xml_path = os.path.join(label_path)

    tree = ET.parse(xml)
    root = tree.getroot()
    obj_xml = root.findall('object')

    result = []
    name_list = []
    idx = 0
    category_id = []

    for obj in obj_xml:
        bbox_original = obj.find('bndbox')
        names = obj.find('name')

        xmin = int(float(bbox_original.find('xmin').text))
        ymin = int(float(bbox_original.find('ymin').text))
        xmax = int(float(bbox_original.find('xmax').text))
        ymax = int(float(bbox_original.find('ymax').text))

        result.append([xmin, ymin, xmax, ymax])
        name_list.append(names.text)
        category_id.append(idx)
        idx+=1
    
    return result, name_list


def crop_image(image, bbox, str_label, output_path):
    idx = 0
    for i in zip(bbox, str_label):
        print(i)
        label = i[1]
        xmin = i[0][0]
        ymin = i[0][1]
        xmax = i[0][2]
        ymax = i[0][3]

        cropped_img = image[ymin:ymax, xmin:xmax]
        cropped_img2 = image[ymin+170:ymax, xmin:xmax]
        cropped_img3 = image[ymin:ymax, xmin+50:xmax]
        cropped_img4 = image[ymin+170:ymax, xmin+50:xmax]

        # cv2.imshow('cropped', cropped_img)
        # cv2.imshow('cropped2', cropped_img2)
        # cv2.imshow('cropped3', cropped_img3)
        # cv2.waitKey(0)

        if not(os.path.isdir(output_path + '/crop_images/' + label)):
            os.makedirs(os.path.join(output_path + '/crop_images/' + label))

        else:
            pass
        
        now = datetime.datetime.now()
        nowdate = now.strftime('%Y-%m-%d %H:%M:%S')
        cv2.imwrite(output_path + '/crop_images/' + label + '/' + label + '_' + str(time.time()) + '_original_' + str(idx) + '.jpg', cropped_img)
        cv2.imwrite(output_path + '/crop_images/' + label + '/' + label + '_' + str(time.time()) + '_hh_' + str(idx) + '.jpg', cropped_img2)
        cv2.imwrite(output_path + '/crop_images/' + label + '/' + label + '_' + str(time.time()) + '_ww_' + str(idx) + '.jpg', cropped_img3)
        cv2.imwrite(output_path + '/crop_images/' + label + '/' + label + '_' + str(time.time()) + '_hw_' + str(idx) + '.jpg', cropped_img4)
        idx+=1    


if __name__ == "__main__":
    image_set_path = sys.argv[1] + '/*'
    image_list = sorted(glob.glob(image_set_path))

    xml_set_path = sys.argv[2] + '/*'
    xml_list = sorted(glob.glob(xml_set_path))

    output_path = sys.argv[3]

    if not(os.path.isdir(output_path)):
        os.makedirs(os.path.join(output_path))

    else:
        pass


    for image, xml in zip(image_list, xml_list):
        print(image, xml)
        
        image_name = image.split('/')[-1]
        image_name = image_name.split('.')[0]

        image = cv2.imread(image)

        bbox, str_label = get_boxes(xml)
        crop_image(image, bbox, str_label, output_path)
        # break
        