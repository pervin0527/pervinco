import pathlib
import cv2
import random
import albumentations as A
import xml.etree.ElementTree as ET
from matplotlib import pyplot as plt


def get_lists(path):
    ds_path = pathlib.Path(path)

    images = list(ds_path.glob('*.jpg'))
    images = sorted([str(path) for path in images])
    
    xmls = list(ds_path.glob('*.xml'))
    xmls = sorted([str(path) for  path in xmls])

    return images, xmls


def visualize(class_dict):
    x, y = list(class_dict.keys()), list(class_dict.values())
    plt.figure(figsize=(10, 10))
    plt.bar(x, y, width=0.9,)
    plt.xticks(x, rotation=90)
    plt.show()


def get_boxes(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    obj_xml = root.findall('object')
    
    if obj_xml[0].find('bndbox') != None:

        result = []
        name_list = []
        idx = 0
        category_id = []

        for obj in obj_xml:
            bbox_original = obj.find('bndbox')
            names = obj.find('name')
        
            xmin = int(float(bbox_original.find('xmin').text))
            ymin = int(float(bbox_original.find('ymin').text))
            xmax = int(float(bbox_original.find('xmax').text))
            ymax = int(float(bbox_original.find('ymax').text))

            result.append([xmin, ymin, xmax, ymax])
            name_list.append(names.text)
            category_id.append(idx)
            idx+=1
        
        return result, name_list, category_id


def read_and_count(xml_list):
    label_map = {}

    for xml in xml_list:
        tree = ET.parse(xml)
        root = tree.getroot()

        object = root.findall('object')
        
        for obj in object:
            class_name = obj.find('name').text
            # print(class_name)

            if class_name not in label_map:
                label_map.update({class_name : 1})

            else:
                label_map[class_name] += 1

    return label_map


def modify_coordinate(augmented, xml, idx):
    filename = xml.split('/')[-1]
    filename = filename.split('.')[0]

    tree = ET.parse(xml)
    root = tree.getroot()

    img_file_name = root.find('filename').text
    img_file_name = f"{img_file_name.split('.')[0]}_{idx}.jpg"

    root.find('filename').text = img_file_name
    
    obj_xml = root.findall('object')
    
    auged_height, auged_width, _ = augmented['image'].shape
    size_xml = root.find('size')
    size_xml.find('width').text = str(int(auged_width))
    size_xml.find('height').text = str(int(auged_height))
    
    bbox_mod = augmented['bboxes']

    for x, obj in enumerate (obj_xml):
        bbox_original = obj.find('bndbox')
        bbox_original.find('xmin').text = str(int(bbox_mod[x][0]))
        bbox_original.find('ymin').text = str(int(bbox_mod[x][1]))
        bbox_original.find('xmax').text = str(int(bbox_mod[x][2]))
        bbox_original.find('ymax').text = str(int(bbox_mod[x][3]))

    tree.write(f"{output_path}/{filename}_{idx}.xml")


def augmentation(image_list, xml_list, info):
    for image_file, xml_file in zip(image_list, xml_list):
        image_file_name = (image_file.split('/')[-1]).split('.')[0]
        image = cv2.imread(image_file)

        bboxes, classes, category_ids = get_boxes(xml_file)
        aug_num = random.randint(1, 30)

        if len(classes) == 1:
            for idx in range(aug_num):
                if check_list[labels.index(classes[0])] + aug_num < maximum:
                    check_list[labels.index(classes[0])] += 1
                    try:
                        transformed = transform(image=image, bboxes=bboxes, category_ids=category_ids)
                        cv2.imwrite(f"{output_path}/{image_file_name}_{idx}.jpg", transformed["image"])
                        modify_coordinate(transformed, xml_file, idx)

                    except:
                        pass

        
if __name__ == "__main__":
    input_path = "/data/Datasets/Seeds/test/inputs"
    output_path = "/data/Datasets/Seeds/test/outputs"
    maximum = 30

    transform = A.Compose([A.RandomRotate90(p=1),

                           A.OneOf([A.RandomBrightness(p=0.6),
                                    A.RandomContrast(p=0.6)], p=0.7),

                           A.OneOf([A.HorizontalFlip(p=0.6),
                                    A.VerticalFlip(p=0.6)], p=0.7),

                           ], bbox_params= A.BboxParams(format='pascal_voc', label_fields=['category_ids']))

    input_images, input_xmls = get_lists(input_path)
    input_info = read_and_count(input_xmls)
    labels = sorted(list(input_info.keys()))
    print(labels)

    check_list = [0] * len(labels)

    # visualize(input_info)
    print(input_info)

    augmentation(input_images, input_xmls, input_info)

    output_images, output_xmls = get_lists(output_path)
    output_info = read_and_count(output_xmls)
    print(output_info)