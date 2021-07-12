import argparse
import os 
import glob
import sys 
import pandas as pd 
import xml.etree.ElementTree as ET
from datetime import datetime
from xml.dom import minidom


def xml_to_csv(path):
    xml_list = []

    for xml_file in sorted(glob.glob(f'{path}/*.xml')):
        # print(xml_file)

        img_file_name = xml_file.split('/')[-1]
        file_name = img_file_name.split('.')[0] + '.jpg'

        if not os.path.isfile(f'{path}/{file_name}'):
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

                xmin = int(float(bbox_original.find('xmin').text))
                ymin = int(float(bbox_original.find('ymin').text))
                xmax = int(float(bbox_original.find('xmax').text))
                ymax = int(float(bbox_original.find('ymax').text))

                value = (str(file_name), height, width, str(name), xmin, ymin, xmax, ymax)
                xml_list.append(value)

    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="model maker data pre-processing")
    parser.add_argument('--dataset_path', type=str)
    args = parser.parse_args()
    
    dataset_name = args.dataset_path.split('/')[-1]

    output_path = args.dataset_path
    output_path = output_path.split('/')[:-1]
    output_path = '/'.join(output_path)

    if not(os.path.isdir(output_path)):
        os.makedirs(os.path.join(output_path))

    xml_df = xml_to_csv(args.dataset_path)
    today = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
    # xml_df.to_csv(output_path + '/' + dataset_name + '_' + today + '_feature.csv', index=None)
    xml_df.to_csv(f'{output_path}/{dataset_name}_{today}_feature.csv', index=None)