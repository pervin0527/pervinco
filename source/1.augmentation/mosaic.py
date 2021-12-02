import cv2
import os
import glob
import random
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from lxml.etree import Element, SubElement, tostring
from PIL import Image


def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def dataset(anno_dir, img_dir):
    img_paths = []
    annos = []
    for anno_file in glob.glob(os.path.join(anno_dir, '*.txt')):
        anno_id = anno_file.split('/')[-1].split('.')[0].split('\\')[-1]

        with open(anno_file, 'r') as f:
            num_of_objs = int(file_len(f.name))

            img_path = os.path.join(img_dir, f'{anno_id}.jpg')
            img = cv2.imread(img_path)
            img_height, img_width, _ = img.shape
            del img

            boxes = []
            for _ in range(num_of_objs):
                obj = f.readline().rstrip().split(' ')
                obj = [float(elm) for elm in obj]
                obj[0] = int(obj[0])
                
                xmin = max(obj[1], 0)
                ymin = max(obj[2], 0)
                xmax = min(obj[3], img_width) 
                ymax = min(obj[4], img_height)

                boxes.append([obj[0], xmin, ymin, xmax, ymax])
                # print(boxes)

            if not boxes:
                continue

        img_paths.append(img_path)
        annos.append(boxes)
    return img_paths, annos


def write_xml(annot_data):
    node_root = Element("annotation")
    
    node_folder = SubElement(node_root, "folder")
    node_folder.text = "images"

    node_filename = SubElement(node_root, "filename")
    node_filename.text = f'output_{VOLUME}.jpg'
    
    node_size = SubElement(node_root, "size")
    node_width = SubElement(node_size, "width")
    node_width.text = str(OUTPUT_SIZE[1])
    node_height = SubElement(node_size, "height")
    node_height.text = str(OUTPUT_SIZE[0])
    node_depth = SubElement(node_size, "depth")
    node_depth.text = "3"

    if len(annot_data) > 0:
        for idx in range(len(annot_data)):
            # print(annot_data, annot_data[1])
            node_object = SubElement(node_root, "object")
            node_name = SubElement(node_object, "name")
            node_name.text = category_name[annot_data[idx][0]]
            
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
            node_xmin.text = str(int(annot_data[idx][1]))
            node_ymin = SubElement(node_bndbox, 'ymin')
            node_ymin.text = str(int(annot_data[idx][2]))
            node_xmax = SubElement(node_bndbox, 'xmax')
            node_xmax.text = str(int(annot_data[idx][3]))
            node_ymax = SubElement(node_bndbox, 'ymax')
            node_ymax.text = str(int(annot_data[idx][4]))

    tree = ET.ElementTree(node_root)
    tree.write(f'{OUTPUT_DIR}/annotations/output_{VOLUME}.xml')


def mosaic(all_img_list, all_annos, idxs, output_size, scale_range, filter_scale=0.):
    output_img = np.zeros([output_size[0], output_size[1], 3], dtype=np.uint8)
    scale_x = scale_range[0] + random.random() * (scale_range[1] - scale_range[0])
    scale_y = scale_range[0] + random.random() * (scale_range[1] - scale_range[0])
    divid_point_x = int(scale_x * output_size[1])
    divid_point_y = int(scale_y * output_size[0])

    new_anno = []
    for i, idx in enumerate(idxs):
        path = all_img_list[idx]
        img_annos = all_annos[idx]

        img = cv2.imread(path)
        if i == 0:  # top-left
            img = cv2.resize(img, (divid_point_x, divid_point_y))
            
            # cv2.imshow('top-left', img)

            output_img[:divid_point_y, :divid_point_x, :] = img
            for bbox in img_annos:
                
                xmin = bbox[1] - bbox[3]*0.5
                ymin = bbox[2] - bbox[4]*0.5
                xmax = bbox[1] + bbox[3]*0.5
                ymax = bbox[2] + bbox[4]*0.5

                xmin *= scale_x
                ymin *= scale_y
                xmax *= scale_x
                ymax *= scale_y
                new_anno.append([bbox[0], xmin, ymin, xmax, ymax])

        elif i == 1:  # top-right
            img = cv2.resize(img, (output_size[1] - divid_point_x, divid_point_y))

            # cv2.imshow('top-right', img)

            output_img[:divid_point_y, divid_point_x:output_size[1], :] = img
            for bbox in img_annos:
                xmin = bbox[1] - bbox[3]*0.5
                ymin = bbox[2] - bbox[4]*0.5
                xmax = bbox[1] + bbox[3]*0.5
                ymax = bbox[2] + bbox[4]*0.5

                xmin = scale_x + xmin * (1 - scale_x)
                ymin = ymin * scale_y
                xmax = scale_x + xmax * (1 - scale_x)
                ymax = ymax * scale_y
                new_anno.append([bbox[0], xmin, ymin, xmax, ymax])

        elif i == 2:  # bottom-left
            img = cv2.resize(img, (divid_point_x, output_size[0] - divid_point_y))

            # cv2.imshow('bottom-left', img)

            output_img[divid_point_y:output_size[0], :divid_point_x, :] = img
            for bbox in img_annos:
                xmin = bbox[1] - bbox[3]*0.5
                ymin = bbox[2] - bbox[4]*0.5
                xmax = bbox[1] + bbox[3]*0.5
                ymax = bbox[2] + bbox[4]*0.5

                xmin = xmin * scale_x
                ymin = scale_y + ymin * (1 - scale_y)
                xmax = xmax * scale_x
                ymax = scale_y + ymax * (1 - scale_y)
                new_anno.append([bbox[0], xmin, ymin, xmax, ymax])


        else:  # bottom-right
            img = cv2.resize(img, (output_size[1] - divid_point_x, output_size[0] - divid_point_y))

            # cv2.imshow('bottom-right', img)

            output_img[divid_point_y:output_size[0], divid_point_x:output_size[1], :] = img
            for bbox in img_annos:
                xmin = bbox[1] - bbox[3]*0.5
                ymin = bbox[2] - bbox[4]*0.5
                xmax = bbox[1] + bbox[3]*0.5
                ymax = bbox[2] + bbox[4]*0.5

                xmin = scale_x + xmin * (1 - scale_x)
                ymin = scale_y + ymin * (1 - scale_y)
                xmax = scale_x + xmax * (1 - scale_x)
                ymax = scale_y + ymax * (1 - scale_y)
                new_anno.append([bbox[0], xmin, ymin, xmax, ymax])

    if 0 < filter_scale:
        new_anno = [anno for anno in new_anno if
                    filter_scale < (anno[3] - anno[1]) and filter_scale < (anno[4] - anno[2])]

    # cv2.waitKey(0)
    return output_img, new_anno


if __name__ == '__main__':
    OUTPUT_SIZE = (640, 480)  
    SCALE_RANGE = (0.3, 0.7)
    FILTER_TINY_SCALE = 1 / 50

    ANNO_DIR = '/data/Datasets/Seeds/SPC/set11/train3/labels'
    IMG_DIR = '/data/Datasets/Seeds/SPC/set11/train3/images'
    LABEL_DIR = '/data/Datasets/Seeds/SPC/Labels/labels.txt'
    OUTPUT_DIR = '/data/Datasets/Seeds/SPC/set11/train3/mosaic'
    VOLUME = 2201

    if not os.path.isdir(OUTPUT_DIR):
        os.makedirs(f'{OUTPUT_DIR}/images')
        os.makedirs(f'{OUTPUT_DIR}/annotations')

    category_name = pd.read_csv(LABEL_DIR, sep=',', index_col=False, header=None)
    category_name = category_name[0].tolist()
    print(category_name)

    img_paths, annos = dataset(ANNO_DIR, IMG_DIR)
    print(len(img_paths), len(annos))

    while VOLUME:
        idxs = random.sample(range(len(annos)), 4)
        new_image, new_annos = mosaic(img_paths, annos,idxs, OUTPUT_SIZE, SCALE_RANGE, filter_scale=FILTER_TINY_SCALE)

        if len(new_annos) == 4:
            cv2.imwrite(f'{OUTPUT_DIR}/images/output_{VOLUME}.jpg', new_image) #The mosaic image
            write_xml(new_annos)

            # for anno in new_annos:
            #     xmin, ymin = int(anno[1] * OUTPUT_SIZE[1]), int(anno[2] * OUTPUT_SIZE[0])
            #     xmax, ymax = int(anno[3] * OUTPUT_SIZE[1]), int(anno[4] * OUTPUT_SIZE[0])
            #     # print(xmin, ymin, xmax, ymax)
            #     cv2.rectangle(new_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1, cv2.LINE_AA)
                
            # cv2.imshow("result", new_image)
            # cv2.waitKey(0)
            
            VOLUME -= 1