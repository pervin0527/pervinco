import os
from glob import glob
import xml.etree.ElementTree as ET

xml_folder = '/data/Datasets/SPC/pb/train/annotations-test'
limit_ratio = 4

for xml_path in glob(os.path.join(xml_folder, '*.xml')):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find('size')
    height = int(size.find('height').text)
    for obj in root.findall('object'):
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        if (xmax - xmin) / (ymax - ymin) > limit_ratio:
            print(f'({xmin}, {ymin}, {xmax}, {ymax}) -> ', end='')
            ymed = (ymax + ymin) // 2
            yhlf = (xmax - xmin) // (2 * limit_ratio)
            ymin = max(0, ymed - yhlf)
            ymax = min(height - 1, ymed + yhlf)
            print(f'({xmin}, {ymin}, {xmax}, {ymax})')
            bbox.find('ymin').text = str(ymin)
            bbox.find('ymax').text = str(ymax)
    tree.write(xml_path)
print('done')