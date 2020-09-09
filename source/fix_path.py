from xml.etree import ElementTree as ET
import glob

path = sorted(glob.glob('./VOCdevkit/VOC2007/Annotations/*.xml'))
print(len(path))
    
for p in path:
    tree = ET.parse(p)

    for xml_path in tree.findall('folder'):
        if xml_path.text != 'VOC2007':
            xml_path.text = 'VOC2007'
            print(xml_path)
    tree.write(p)