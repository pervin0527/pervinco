import os
import cv2
import sys
import pandas as pd
from glob import glob
import xml.etree.ElementTree as ET
from lxml.etree import Element, SubElement, tostring

def get_annot_data(txt_df):
    annot = []
    for data in txt_df:
        data = data.split(' ')
        label, xmin, ymin, xmax, ymax = label_map[int(data[0])], int(data[1]), int(data[2]), int(data[3]), int(data[4])
        annot.append((label, xmin, ymin, xmax, ymax))

    return annot

def write_xml(annot_data, filename, height, width):
    node_root = Element("annotation")
    
    node_folder = SubElement(node_root, "folder")
    node_folder.text = "images"

    node_filename = SubElement(node_root, "filename")
    node_filename.text = f'{filename}.jpg'
    
    node_size = SubElement(node_root, "size")
    node_width = SubElement(node_size, "width")
    node_width.text = str(width)
    node_height = SubElement(node_size, "height")
    node_height.text = str(height)
    node_depth = SubElement(node_size, "depth")
    node_depth.text = "3"

    if len(annot_data) > 0:
        for idx in range(len(annot_data)):
            # print(annot_data, annot_data[1])
            node_object = SubElement(node_root, "object")
            node_name = SubElement(node_object, "name")
            node_name.text = str(annot_data[idx][0])
            
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
    tree.write(f'{root}/annotations/{filename}.xml')

if __name__ == "__main__":
    root = "/data/Datasets/Seeds/SPC/set10"
    label_file = "/data/Datasets/Seeds/SPC/Labels/labels.txt"

    try:
        if not os.path.isdir(f'{root}/annotations') or os.path.isdir(f'{root}/images'):
            os.makedirs(f"{root}/annotations")
            os.makedirs(f"{root}/images")

    except:
        pass

    label_file = pd.read_csv(label_file, sep=',', index_col=False, header=None)
    label_map = label_file[0].tolist()

    # files = sorted(glob(f"{root}/frames/*/*.jpg"))
    files = sorted(glob(f"{root}/images/*.jpg"))

    for count, file in enumerate(files):
        filename = file.split('/')[-1].split('.')[0]
        # foldername = file.split('/')[-2]

        # txt_file = f"{root}/detect/{foldername}/labels/{filename}.txt"
        txt_file = f"{root}/labels/{filename}.txt"

        # if os.path.isfile(f"{root}/detect/{foldername}/labels/{filename}.txt"):
        if os.path.isfile(f"{root}/labels/{filename}.txt"):
            df = pd.read_csv(txt_file, sep=',', header=None, index_col=False)
            df_list = df[0].tolist()

            annot_data = get_annot_data(df_list)
            # print(annot_data)

            image = cv2.imread(file)
            img_height, img_width = image.shape[:-1]

            # for data in annot_data:
            #     cv2.rectangle(image, (data[1], data[2]), (data[3], data[4]), (0, 0, 255), thickness=2)
            #     cv2.putText(image, data[0], (data[1], data[2]), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255))
            # cv2.imshow('result', image)
            # cv2.waitKey(0)
            # break

            os.system("clear")
            print(count)
            write_xml(annot_data, filename, img_height, img_width)
            cv2.imwrite(f'{root}/images/{filename}.jpg', image)