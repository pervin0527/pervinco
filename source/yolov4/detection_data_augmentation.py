import random
import cv2
import os
import sys
import glob
from matplotlib import pyplot as plt
import xml.etree.ElementTree as ET
from xml.dom import minidom
import albumentations as A

BOX_COLOR = (0, 255, 0) # Red
TEXT_COLOR = (255, 255, 255) # White


def visualize_bbox(img, bbox, class_name, color=BOX_COLOR, thickness=2):   
    # x_min, y_min, w, h = bbox
    # x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)
    x_min, y_min, x_max, y_max = map(lambda x : int(x), bbox)
    
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)    
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35, 
        color=TEXT_COLOR, 
        lineType=cv2.LINE_AA,
    )
    return img


def visualize(image, bboxes, category_ids, category_id_to_name):
    img = image.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        img = visualize_bbox(img, bbox, class_name)
    # plt.figure(figsize=(12, 12))
    # plt.axis('off')
    # plt.imshow(img)

    cv2.imshow('sample', img)
    cv2.waitKey(0)


def get_boxes(label_path):
    tree = ET.parse(xml)
    root = tree.getroot()
    obj_xml = root.findall('object')
    
    if obj_xml[0].find('bndbox') != None:

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

            # print(result)
            # print(name_list)
            # print(category_id)
        
        return result, name_list, category_id


def make_categori_id(str_label):
    idx = 0
    category_id_to_name = {}

    for label in str_label:
        category_id_to_name.update({int(idx):str(label)})
        idx += 1

    return category_id_to_name


def modify_coordinate(output_path, augmented, xml, idx):
    filename = xml.split('/')[-1]
    filename = filename.split('.')[0]

    tree = ET.parse(xml)
    root = tree.getroot()
    obj_xml = root.findall('object')
    
    auged_width, auged_height, _ = augmented['image'].shape
    size_xml = root.find('size')
    size_xml.find('width').text = str(int(auged_width))
    size_xml.find('height').text = str(int(auged_height))
    
    bbox_mod = augmented['bboxes']

    for x, obj in enumerate (obj_xml):
        bbox_original = obj.find('bndbox')
        bbox_original.find('xmin').text = str(int(bbox_mod[x][0]))
        bbox_original.find('ymin').text = str(int(bbox_mod[x][1]))
        bbox_original.find('xmax').text = str(int(bbox_mod[x][2]))
        bbox_original.find('ymax').text = str(int(bbox_mod[x][3]))

        # del bbox_mod[0]

    root.find('filename').text = filename + '_' + str(idx) + '.jpg'

    tree.write(output_path + '/' + str(idx) + '/xmls/' + filename + '_' + str(idx) + '.xml')



if __name__ == "__main__":
    image_set_path = sys.argv[1] + '/*'
    image_list = sorted(glob.glob(image_set_path))

    xml_set_path = sys.argv[2] + '/*'
    xml_list = sorted(glob.glob(xml_set_path))

    output_path = sys.argv[3]

    aug_num = sys.argv[4]

    for x in range(int(aug_num)):
        if not(os.path.isdir(output_path + '/' + str(x))):
            os.makedirs(os.path.join(output_path + '/' + str(x) + '/images'))
            os.makedirs(os.path.join(output_path + '/' + str(x) + '/xmls'))

        else:
            pass

    for image, xml in zip(image_list, xml_list):        
        image_name = image.split('/')[-1]
        image_name = image_name.split('.')[0]

        xml_name = xml.split('/')[-1]

        image = cv2.imread(image)
        height, width, channels = image.shape

        bbox, str_label, category_id = get_boxes(xml)
        category_id_to_name = make_categori_id(str_label)

        # visualize(image, bbox, category_id, category_id_to_name)

        print(image_name, xml_name, str_label)

        transform = A.Compose([
            A.Resize(608, 608, p=1),
            A.HorizontalFlip(p=1),
            # A.ShiftScaleRotate(p=0.5, border_mode=1),
            A.RandomRotate90(p=0.5),
            A.Blur(p=0.3),
            A.HorizontalFlip(p=0.6),
            A.VerticalFlip(p=0.3),
            A.RandomBrightnessContrast(p=0.3),
            A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=0.3),
            ],
            bbox_params = A.BboxParams(format='pascal_voc', label_fields=['category_ids'])
        )

        for x in range(int(aug_num)):
            transformed = transform(image=image, bboxes=bbox, category_ids=category_id)

            # visualize(transformed['image'], transformed['bboxes'], transformed['category_ids'], category_id_to_name,)
            cv2.imwrite(output_path + '/' + str(x) + '/images/' + image_name + '_' + str(x) + '.jpg', transformed['image'])
            
            modify_coordinate(output_path, transformed, xml, x)