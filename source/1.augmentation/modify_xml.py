import cv2
import pathlib
import xml.etree.ElementTree as ET

from xml.dom import minidom

def modify_xml(images, xmls, is_train):
    if is_train:
        output_path = f'{PATH}/train2'

    else:
        output_path = f'{PATH}/test2'

    for xml in xmls:
        file_name = xml.split('/')[-1]
        tree = ET.parse(xml)
        root = tree.getroot()
        obj_xml = root.findall('annotation')

        img_file_name = root.find('filename').text
        root.find('filename').text = img_file_name.split('.')[0] + '.jpg'
        tree.write(f'{output_path}/{file_name}')

    for image in images:
        img_name = image.split('/')[-1]
        img_name = img_name.split('.')[0]
        img = cv2.imread(image)
        cv2.imwrite(f'{output_path}/{img_name}.jpg', img)


def get_lists(ds_path):
    images = list(ds_path.glob('*.png'))
    images = [str(path) for path in images]

    xmls = list(ds_path.glob('*.xml'))
    xmls = [str(path) for path in xmls]

    return images, xmls


if __name__ == "__main__":
    PATH = "/data/Datasets/Seeds/mm_data"

    train_path = pathlib.Path(f'{PATH}/train')
    test_path = pathlib.Path(f'{PATH}/test')

    train_images, train_xmls = get_lists(train_path)
    test_images, test_xmls = get_lists(test_path)

    modify_xml(train_images, train_xmls, True)
    modify_xml(test_images, test_xmls, False)