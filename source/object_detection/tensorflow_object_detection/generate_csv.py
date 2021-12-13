import os 
import glob
import sys 
import argparse
import pandas as pd 
import xml.etree.ElementTree as ET
from datetime import datetime
from xml.dom import minidom


def xml_to_csv(path):
    xml_list = []

    for xml_file in sorted(glob.glob(path + '/*.xml')):
        img_file_name = xml_file.split('/')[-1]
        file_name = img_file_name.split('.')[0] + '.jpg'

        if not os.path.isfile(f'{image_path}/{file_name}'):
            print(file_name)
            break

        tree = ET.parse(xml_file)
        root = tree.getroot()

        obj_xml = root.findall('object')
        size_xml = root.findall('size')
        # file_name = root.find('filename').text

        for size in size_xml:
            height = int(size.find('height').text)
            width = int(size.find('width').text)

        for obj in obj_xml:
            if obj.find('bndbox') != None:
                bbox_original = obj.find('bndbox')
                
                name = obj.find('name').text
                
                if name == "Etc":
                    pass

                xmin = int(bbox_original.find('xmin').text)
                ymin = int(bbox_original.find('ymin').text)
                xmax = int(bbox_original.find('xmax').text)
                ymax = int(bbox_original.find('ymax').text)

                value = (str(file_name), height, width, str(name), xmin, ymin, xmax, ymax)
                xml_list.append(value)

    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate csv file from xml annotations')
    parser.add_argument('--annot_dir', help='directory of input xml files', default='../data/xml_annot')
    parser.add_argument('--out_csv_path', help='path to output csv file', default='../data/train.csv')
    args = parser.parse_args()

    xml_path = args.annot_dir
    dataset_name = xml_path.split('/')[-2]
    image_path = xml_path.split('/')[:-1]
    output_path = '/'.join(image_path)
    image_path = '/'.join(image_path) + "/images"

    if not(os.path.isdir(output_path)):
        os.makedirs(os.path.join(output_path))

    else:
        pass

    xml_df = xml_to_csv(xml_path)
    xml_df.to_csv(args.out_csv_path, index=None)