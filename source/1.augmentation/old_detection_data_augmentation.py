import cv2
import os
import sys
import glob
import random
import argparse
import albumentations as A
import xml.etree.ElementTree as ET
from xml.dom import minidom
from matplotlib import pyplot as plt
from tqdm import tqdm

BOX_COLOR = (0, 0, 255) # Red
TEXT_COLOR = (255, 255, 255) # White
# TEXT_COLOR = (0, 0, 0)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def visualize_bbox(img, bbox, class_name, color=BOX_COLOR, thickness=3):   
    # x_min, y_min, w, h = bbox
    # x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)
    x_min, y_min, x_max, y_max = map(lambda x : int(x), bbox)
    
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)    
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), (0, 0, 0), -1)
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


def visualize(image, bboxes, category_ids, category_id_to_name, window_name):
    img = image.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        print(bbox, class_name)
        img = visualize_bbox(img, bbox, class_name)

    img = cv2.resize(img, (720, 540))
    cv2.imshow(str(window_name), img)
    cv2.waitKey(0)


def get_boxes(label_path):
    tree = ET.parse(label_path)
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
        
        return result, name_list, category_id


def make_categori_id(str_label):
    idx = 0
    category_id_to_name = {}

    for label in str_label:
        category_id_to_name.update({int(idx):str(label)})
        idx += 1

    return category_id_to_name


def modify_coordinate(output_path, augmented, xml, idx, output_shape):
    filename = xml.split('/')[-1]
    filename = filename.split('.')[0]

    tree = ET.parse(xml)
    root = tree.getroot()
    obj_xml = root.findall('object')
    
    auged_height, auged_width, _ = augmented['image'].shape
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

    root.find('filename').text = f"{filename}_{str(idx)}.jpg"

    if output_shape == 'split':
        tree.write(f"{output_path}/{str(idx)}/annotations/{filename}_{str(idx)}.xml")

    else:
        tree.write(f"{output_path}/annotations/{filename}_{str(idx)}.xml")


def augmentation(image_list, xml_list, output_shape, visual):
    cnt = 0
    for i in tqdm(range(len(image_list))):
        image = image_list[i]
        xml = xml_list[i]        
        image_name = image.split('/')[-1]
        image_name = image_name.split('.')[0]

        xml_name = xml.split('/')[-1]

        image = cv2.imread(image)
        height, width, channels = image.shape

        try:
            bbox, str_label, category_id = get_boxes(xml)
            category_id_to_name = make_categori_id(str_label)
            # print(image_name, xml_name, str_label)

            if visual:
                visualize(image, bbox, category_id, category_id_to_name, 'original data')

            transform = A.Compose([
                A.OneOf([
                    A.Resize(448, 448, p=1),
                    A.RandomSizedBBoxSafeCrop(448, 448, p=1),
                ], p=1),


                A.OneOf([
                    A.Rotate(p=1, border_mode=0, limit=(-15, 15)),
                    A.ShiftScaleRotate(rotate_limit=(-15, 15), border_mode=0),
                ], p=1),

                # A.RandomBrightnessContrast(brightness_limit=(-0.15, 0.15), contrast_limit=(-0.15, 0.15), p=1),
                A.RandomBrightness(limit=(-0.15, 0.15),p=1)

                ], bbox_params = A.BboxParams(format='pascal_voc', label_fields=['category_ids']))

            if output_shape == 'split':
                for x in range(int(aug_num)):
                    if os.path.isdir(f"{output_path}/{str(x)}/images") and os.path.isdir(f"{output_path}/{str(x)}/annotations"):
                        pass
                    else:
                        os.makedirs(f"{output_path}/{str(x)}/images")
                        os.makedirs(f"{output_path}/{str(x)}/annotations")

                    transformed = transform(image=image, bboxes=bbox, category_ids=category_id)
                    cv2.imwrite(f"{output_path} /{str(x)}/images/{image_name}_{str(x)}.jpg", transformed['image'])
                    modify_coordinate(output_path, transformed, xml, x, output_shape)

                    if visual:
                        visualize(transformed['image'], transformed['bboxes'], transformed['category_ids'], category_id_to_name, 'augmentation data')

            elif output_shape == 'merge':
                for x in range(int(aug_num)):
                    transformed = transform(image=image, bboxes=bbox, category_ids=category_id)
                    cv2.imwrite(f"{output_path}/images/{image_name}_{str(x)}.jpg", transformed['image'])
                    modify_coordinate(output_path, transformed, xml, x, output_shape)

                    if visual:
                        visualize(transformed['image'], transformed['bboxes'], transformed['category_ids'], category_id_to_name, 'augmentation data')

            else:
                print('Please Select the output format option from "split" or "merge".')

        except:
            print(cnt, xml_name, ' This file does not contain objects.')
            visualize(image, bbox, category_id, category_id_to_name, 'failed data')
            cnt += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Detection dataset augmentation')
    parser.add_argument('--image_path', type=str)
    parser.add_argument('--xml_path', type=str)
    parser.add_argument('--outpath', type=str)
    parser.add_argument('--outshape', type=str) # split or merge
    parser.add_argument('--num_aug', type=str, default=5)
    parser.add_argument('--visual', type=str2bool, default=False)
    args = parser.parse_args()

    image_set_path = f'{args.image_path}/*.jpg'
    image_list = sorted(glob.glob(image_set_path))

    xml_set_path = f'{args.xml_path}/*.xml'
    xml_list = sorted(glob.glob(xml_set_path))
    print(len(image_list), len(xml_list))

    output_shape = args.outshape
    output_path = args.outpath
    aug_num = args.num_aug
    visual = args.visual

    if os.path.isdir(output_path):
        if os.path.isdir(f'{output_path}/images') and os.path.isdir(f'{output_path}/annotations'):
            pass
        else:
            os.makedirs(f'{output_path}/images')
            os.makedirs(f'{output_path}/annotations')
    else:
        os.makedirs(output_path)
        os.makedirs(f'{output_path}/images')
        os.makedirs(f'{output_path}/annotations')


    augmentation(image_list, xml_list, output_shape, visual)