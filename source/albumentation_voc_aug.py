import os
import numpy as np
import cv2
import glob
import sys
import xml.etree.ElementTree as ET
from xml.dom import minidom

from albumentations import(
    BboxParams,
    HorizontalFlip,
    ShiftScaleRotate,
    VerticalFlip,
    Resize,
    CenterCrop,
    RandomCrop,
    Crop,
    Compose,
    RandomContrast,
    normalize_bboxes,
    RandomBrightness,
    IAASharpen,
    MotionBlur,
    OneOf)

BOX_COLOR = (255, 0, 0)
TEXT_COLOR = (255, 255, 255)

def read_image(img_path):
    image = cv2.imread(img_path)
    height, width, channels = image.shape
    # print(height, width)

    return image, height, width


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

        xmin = int(bbox_original.find('xmin').text)
        ymin = int(bbox_original.find('ymin').text)
        xmax = int(bbox_original.find('xmax').text)
        ymax = int(bbox_original.find('ymax').text)

        result.append([xmin, ymin, xmax, ymax])
        name_list.append(names.text)
        category_id.append(idx)
        idx+=1

        # print(result)
        # print(name_list)
        # print(category_id)
    
    return result, name_list, category_id


def visualize_bbox(img, bbox, class_id, class_idx_to_name, color=BOX_COLOR, thickness=2):
    x_min, y_min, x_max, y_max = map(lambda v: int(v), bbox)
    # x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
    class_name = class_idx_to_name[class_id]
    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)    
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(img, class_name, (x_min, y_min - int(0.3 * text_height)), cv2.FONT_HERSHEY_SIMPLEX, 0.35,TEXT_COLOR, lineType=cv2.LINE_AA)
    return img


def visualize(annotations, category_id_to_name):
    img = annotations['image'].copy()
    for idx, bbox in enumerate(annotations['bboxes']):
        img = visualize_bbox(img, bbox, annotations['category_id'][idx], category_id_to_name)

    resized = cv2.resize(img, (1920, 1080))
    cv2.imshow('test', resized)
    cv2.waitKey(0)


def get_aug(min_area=0., min_visibility=0.):
    return Compose([

        # Resize(height=1080, width=1920, p=1),
        # RandomCrop(p=0.2, height=1080, width=1400),
        RandomContrast(p=0.5, limit=(-0.5, 0.3)),
        RandomBrightness(p=0.5, limit=(-0.2, 0.3)),
        HorizontalFlip(p=0.5),

        # OneOf([
        # RandomContrast(p=0.3, limit=(-0.5,1)),
        # RandomBrightness(p=0.3, limit=(-0.2,0.1)),
        # HorizontalFlip(p=0.4),
        # ], p=1),

        # OneOf([
        # RandomCrop(p=0.6, height=1080, width=1280),
        # # CenterCrop(height=1280, width=1080, p=0.9),
        # # ShiftScaleRotate(p=0.7),
        # ], p=0.2),

        ],
        bbox_params=BboxParams(format='pascal_voc', min_area=min_area, 
                               min_visibility=min_visibility, label_fields=['category_id'])
                               
    )


def make_categori_id(str_label):
    idx = 0
    result = []
    category_id_to_name = {}

    for label in str_label:
        category_id_to_name.update({int(idx):str(label)})
        idx += 1

    # print(category_id_to_name)
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


    for image, xml in zip(image_list, xml_list):
        print(image, xml)
        
        image_name = image.split('/')[-1]
        image_name = image_name.split('.')[0]
        # print(image_name)

        image, height, width = read_image(image)
        bbox, str_label, category_id = get_boxes(xml)
        category_id_to_name = make_categori_id(str_label)
        # print(category_id_to_name)
        # print(bbox)

        # bbox = normalize_bboxes(bbox, rows=height, cols=width)

        annotations = {'image':image, 'bboxes':bbox, 'category_id':category_id}
        visualize(annotations, category_id_to_name)

        aug = get_aug()
        for i in range(5):
            try:
                augmented = aug(**annotations)
                visualize(augmented, category_id_to_name)
                cv2.imwrite(output_path + '/images/' + image_name + '_' + str(i) + '.jpg', augmented['image'])
                modify_coordinate(output_path, augmented, xml, i)
                # print(i)


            except Exception as e:
                print(e)
            
            # except:
            #     pass

            # aug = get_aug()
            # augmented = aug(**annotations)
            # visualize(augmented, category_id_to_name)
            # cv2.imwrite(output_path + '/images/' + image_name + '_' + str(i) + '.jpg', augmented['image'])
            # modify_coordinate(output_path, augmented, xml, i)