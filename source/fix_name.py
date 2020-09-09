from xml.etree import ElementTree as ET
import glob

path = sorted(glob.glob('./test/*.xml'))
print(len(path))

for p in path:
    tree = ET.parse(p)

    for name in tree.iter('object'):
        original_name = name.find('name').text
        if original_name != 'product':
            name.find('name').text = 'product'

    tree.write(p)