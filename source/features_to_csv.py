import os 
import glob
import sys 
import pandas as pd 
import xml.etree.ElementTree as ET
from xml.dom import minidom


def xml_to_csv(path):
    xml_list = []

    for xml_file in sorted(glob.glob(path + '/*.xml')):
        print(xml_file)

        tree = ET.parse(xml_file)
        root = tree.getroot()

        obj_xml = root.findall('object')
        size_xml = root.findall('size')
        file_name = root.find('filename').text

        for size in size_xml:
            height = int(size.find('height').text)
            width = int(size.find('width').text)

        for obj in obj_xml:
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
    xml_path = sys.argv[1]
    output_path = sys.argv[2]

    if not(os.path.isdir(output_path)):
        os.makedirs(os.path.join(output_path))

    else:
        pass

    xml_df = xml_to_csv(xml_path)
    xml_df.to_csv(output_path + '/coco2017_feature.csv', index=None)