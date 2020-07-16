import os 
import glob 
import pandas as pd 
import xml.etree.ElementTree as ET
from xml.dom import minidom


def xml_to_csv(path):
    xml_list = []
    for xml_file in sorted(glob.glob(path + '/*.xml')):
        print(xml_file)
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     str(member[0].text),
                     int(float(member[4][0].text)),
                     int(float(member[4][1].text)),
                     int(float(member[4][2].text)),
                     int(float(member[4][3].text))
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main(): 
    xml_path = './beverage/Augmentations/eval_xmls'
    xml_df = xml_to_csv(xml_path)
    xml_df.to_csv('./beverage/csv/eval_labels.csv', index=None)


main()