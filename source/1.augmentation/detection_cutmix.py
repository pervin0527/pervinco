import os
import cv2
import pathlib
import xml.etree.ElementTree as ET
import albumentations as A
from random import choice, randint
from lxml.etree import Element, SubElement

def get_file_list(path):
    dataset = pathlib.Path(path)
    dataset = list(dataset.glob('*'))
    dataset = sorted([str(data) for data in dataset])

    return dataset

def get_annotation_data(annotation_file):
    tree = ET.parse(annotation_file)
    root = tree.getroot()

    objects = root.findall('object')
    
    if len(objects) != 0:
        data = []
        for object in objects:
            label = object.find('name').text

            bbox = object.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)

            data.append((label, [xmin, ymin, xmax, ymax]))

        return data

def get_crop_area(image_path, annotation_path):
    # print(image_path)
    image = cv2.imread(image_path)
    annotation = get_annotation_data(annotation_path)

    objects = []
    for label, bbox in annotation:
        size = randint(150, 512)
        transform = A.Compose([A.Crop(x_min=bbox[0], y_min=bbox[1], x_max=bbox[2], y_max=bbox[3]),
                               A.Resize(height=size, width=size),
                               
                               A.OneOf([A.HorizontalFlip(p=0.6),
                                        A.VerticalFlip(p=0.6)], p=0.8),

                               A.OneOf([A.Cutout(num_holes=1, max_h_size=int(size/2), max_w_size=int(size/2), p=0.5),
                                        A.Downscale(p=0.3)], p=0.8),
                               ])
        crop_object = transform(image=image)['image']

        objects.append((label, crop_object))

    return objects

def write_new_xml(org_data, name, index, width, height):
    node_root = Element("annotation")
    
    node_folder = SubElement(node_root, 'folder')
    node_folder.text = output_path.split('/')[-1]

    node_filename = SubElement(node_root, 'filename')
    node_filename.text = f'{name}_{index}.jpg'
    print(f'{name}_{index}')

    node_size = SubElement(node_root, 'size')

    node_width = SubElement(node_size, 'width')
    node_width.text = str(width)

    node_height = SubElement(node_size, 'height')
    node_height.text = str(height)

    node_depth = SubElement(node_size, 'depth')
    node_depth.text = '3'
    
    if org_data != None:
        for i in range(len(org_data)):
            node_object = SubElement(node_root, 'object')
            node_name = SubElement(node_object,'name')
            node_name.text = str(org_data[i][0])

            node_pose = SubElement(node_object, 'pose')
            node_pose.text = 'Unspecified'

            node_truncated = SubElement(node_object, 'truncated')
            node_truncated.text = '0'

            node_occluded = SubElement(node_object, 'occluded')
            node_occluded.text = '0'

            node_difficult = SubElement(node_object, 'difficult')
            node_difficult.text = '0'

            node_bndbox = SubElement(node_object, 'bndbox')
            node_xmin = SubElement(node_bndbox, 'xmin')
            node_xmin.text = str(org_data[i][1])
            node_ymin = SubElement(node_bndbox, 'ymin')
            node_ymin.text = str(org_data[i][2])
            node_xmax = SubElement(node_bndbox, 'xmax')
            node_xmax.text = str(org_data[i][3])
            node_ymax = SubElement(node_bndbox, 'ymax')
            node_ymax.text = str(org_data[i][4])

    tree = ET.ElementTree(node_root)
    tree.write(f'{output_path}/annotations/{name}_{index}.xml')

def attach_fg_objects(crops, bg_image, number):
    bg_img_filename = (bg_image.split('/')[-1]).split('.')[0]

    bg_image = cv2.imread(bg_image)
    bg_height, bg_width, _ = bg_image.shape

    x_set = set()
    y_set = set()
    annot_data = []
    for index, fg_object in enumerate(crops):
        label = fg_object[0]
        fg_image = fg_object[1]
        fg_height, fg_width, _ = fg_image.shape

        start_x = choice([x for x in range(0, bg_width - fg_width) if x not in x_set])
        start_y = choice([y for y in range(0, bg_height - fg_height) if y not in y_set])
        end_x = start_x + fg_width
        end_y = start_y + fg_height

        if all(x not in x_set for x in range(start_x, end_x)) and all(y not in y_set for y in range(start_y, end_y)):
            for x in range(start_x, end_x):
                x_set.add(x)

            for y in range(start_y, end_y):
                y_set.add(y)

            bg_image[start_y:end_y, start_x:end_x] = fg_image
            annot_data.append((label, start_x, start_y, end_x, end_y))

        cv2.imwrite(f'{output_path}/images/{bg_img_filename}_{number}.jpg', bg_image)
        write_new_xml(annot_data, bg_img_filename, number, bg_width, bg_height)

def show_result(images, annotations):
    for image, annotation in zip(images, annotations):
        image = cv2.imread(image)
        annotation = get_annotation_data(annotation)
        
        for object in annotation:
            label = object[0]
            xmin, ymin, xmax, ymax = object[1][0], object[1][1], object[1][2], object[1][3]

        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 255, 0))
    
        cv2.imshow('result', image)
        cv2.waitKey(0)
            
if __name__ == "__main__":
    images_path = "/data/Datasets/Seeds/ETRI_detection2/train/images"
    annotations_path = "/data/Datasets/Seeds/ETRI_detection2/train/annotations/"
    backgrounds_path = "/home/barcelona/test_code/backgrounds"
    output_path = "/home/barcelona/test_code/output"

    if not os.path.isdir(output_path):
        os.makedirs(f'{output_path}/images')
        os.makedirs(f'{output_path}/annotations')

    images_list = get_file_list(images_path)
    annotations_list = get_file_list(annotations_path)
    background_list = get_file_list(backgrounds_path)

    for index, (image_file, annotation_file) in enumerate(zip(images_list, annotations_list)):
        crop_objects = get_crop_area(image_file, annotation_file)
        bg_image = background_list[(randint(0, len(background_list)-1))]
        attach_fg_objects(crop_objects, bg_image, index)

    output_images = get_file_list(f'{output_path}/images')
    output_annotations = get_file_list(f'{output_path}/annotations')
    show_result(output_images, output_annotations)