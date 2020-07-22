import random
import cv2
import os
import sys
import glob
from matplotlib import pyplot as plt
import xml.etree.ElementTree as ET
from xml.dom import minidom

from albumentations import(BboxParams, HorizontalFlip, ShiftScaleRotate, VerticalFlip, Resize,
                           CenterCrop, RandomCrop, Crop, Compose, RandomContrast,
                           RandomBrightness, IAASharpen, MotionBlur, OneOf, normalize_bboxes, 
                           denormalize_bboxes)

BOX_COLOR = (0, 0, 255) # Red
TEXT_COLOR = (255, 255, 255) # White


def modify_coordinate(output_path, augmented, xml, idx):
    filename = xml.split('/')[-1]
    filename = filename.split('.')[0]

    tree = ET.parse(xml)
    root = tree.getroot()
    obj_xml = root.findall('object')
    
    bbox_mod = augmented['bboxes']
    # print(bbox_mod)

    for obj in obj_xml:
        bbox_original = obj.find('bndbox')
        bbox_original.find('xmin').text = str(int(bbox_mod[0][0]))
        bbox_original.find('ymin').text = str(int(bbox_mod[0][1]))
        bbox_original.find('xmax').text = str(int(bbox_mod[0][2]))
        bbox_original.find('ymax').text = str(int(bbox_mod[0][3]))

        del bbox_mod[0]

    root.find('filename').text = filename + '_' + str(idx) + '.jpg'

    tree.write(output_path + '/xmls/' + filename + '_' + str(idx) + '.xml')


def visualize_bbox(img, bbox, class_name, color=BOX_COLOR, thickness=2):
    x_min, y_min, x_max, y_max = map(lambda v: int(v), bbox)
    # print(x_min, y_min, x_max, y_max)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
    
    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)    
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(img, text=class_name, org=(x_min, y_min - int(0.3 * text_height)), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.35, color=TEXT_COLOR, lineType=cv2.LINE_AA, )
    return img


def visualize(image, bboxes, category_ids, category_id_to_name):
    img = image.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        img = visualize_bbox(img, bbox, class_name)

    img = cv2.resize(img, (1080, 720))
    cv2.imshow('sample', img)
    cv2.waitKey(0)


def get_boxes(label_path):
    xml_path = os.path.join(label_path)

    root_1 = minidom.parse(xml_path)
    bnd_1 = root_1.getElementsByTagName('bndbox')
    names = root_1.getElementsByTagName('name')
    
    result = []
    name_list = []
    category_id = []

    for i in range(len(bnd_1)):
        xmin = int(float(bnd_1[i].childNodes[1].childNodes[0].nodeValue))
        ymin = int(float(bnd_1[i].childNodes[3].childNodes[0].nodeValue))
        xmax = int(float(bnd_1[i].childNodes[5].childNodes[0].nodeValue))
        ymax = int(float(bnd_1[i].childNodes[7].childNodes[0].nodeValue))

        result.append([xmin,ymin,xmax,ymax])
        name_list.append(names[i].childNodes[0].nodeValue)
        category_id.append(i)
    
    return result, name_list, category_id


def make_categori_id(str_label):
    idx = 0
    result = []
    category_id_to_name = {}

    for label in str_label:
        category_id_to_name.update({int(idx):str(label)})
        idx += 1

    return category_id_to_name


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

    if not(os.path.isdir(output_path + '/images')) and not(os.path.isdir(output_path + '/xml')):
        os.makedirs(os.path.join(output_path + '/images'))
        os.makedirs(os.path.join(output_path + '/xmls'))

    else:
        pass


    transform = Compose([
        HorizontalFlip(p=0.6),
        RandomContrast(p=0.5, limit=(-0.5, 0.3)),
        RandomBrightness(p=0.5, limit=(-0.2, 0.5)),
        ],
        bbox_params=BboxParams(format='pascal_voc', label_fields=['category_ids']))
        

    for image, xml in zip(image_list, xml_list):
        print(image, xml)
        
        image_name = image.split('/')[-1]
        image_name = image_name.split('.')[0]

        image = cv2.imread(image)
        height, width, channels = image.shape

        bbox, str_label, category_id = get_boxes(xml)
        # print(bbox, str_label, category_id)
        category_id_to_name = make_categori_id(str_label)
        
        # bbox = denormalize_bboxes(bbox, rows=height, cols=width)
        visualize(image, bbox, category_id, category_id_to_name)

        for i in range(5):
            transformed = transform(image=image, bboxes=bbox, category_ids=category_id)
            visualize(transformed['image'], transformed['bboxes'], transformed['category_ids'], category_id_to_name)
            cv2.imwrite(output_path + '/images/' + image_name + '_' + str(i) + '.jpg', transformed['image'])
            modify_coordinate(output_path, transformed, xml, i)