import pathlib
import cv2
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom
from lxml.etree import Element, SubElement, tostring
from sklearn.model_selection import train_test_split


def read_annot_file(path):
    # print(path)
    if os.path.isfile(path):
        tree = ET.parse(path)
        root = tree.getroot()
        objects = root.findall('object')

        annot_data = []
        if objects[0].find('bndbox') != None:
            for obj in objects:
                bboxes = obj.find('bndbox')
                names = obj.find('name')
                CLASSES.add(names.text)
                
                xmin = int(float(bboxes.find('xmin').text))
                ymin = int(float(bboxes.find('ymin').text))
                xmax = int(float(bboxes.find('xmax').text))
                ymax = int(float(bboxes.find('ymax').text))

                annot_data.append((names.text, xmin, ymin, xmax, ymax))

            # print(annot_data)
        return annot_data

    else:
        return None


def write_new_xml(org_data, save_path, index, width, height):
    node_root = Element("annotation")
    
    node_folder = SubElement(node_root, 'folder')
    node_folder.text = save_path.split('/')[-1]

    node_filename = SubElement(node_root, 'filename')
    node_filename.text = f'{index}.jpg'

    node_size = SubElement(node_root, 'size')

    node_width = SubElement(node_size, 'width')
    node_width.text = str(width)

    node_height = SubElement(node_size, 'height')
    node_height.text = str(height)

    node_depth = SubElement(node_size, 'depth')
    node_depth.text = '3'
    
    if org_data != None:
        for i in range(len(org_data)):
            node_object = SubElement(node_root, 'object')
            node_name = SubElement(node_object,'name')
            node_name.text = str(org_data[i][0])

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
            node_xmin.text = str(org_data[i][1])
            node_ymin = SubElement(node_bndbox, 'ymin')
            node_ymin.text = str(org_data[i][2])
            node_xmax = SubElement(node_bndbox, 'xmax')
            node_xmax.text = str(org_data[i][3])
            node_ymax = SubElement(node_bndbox, 'ymax')
            node_ymax.text = str(org_data[i][4])

    tree = ET.ElementTree(node_root)
    tree.write(f'{save_path}/annotations/{index}.xml')
    print(f'{save_path}/annotations/{index}.xml')

    
def process(image_list, is_train):
    if is_train:
        output_dir = f'{output_path}/train'

        if not os.path.isdir(output_dir):
            os.makedirs(f'{output_dir}/images')
            os.makedirs(f'{output_dir}/annotations')

    else:
        output_dir = f'{output_path}/valid'

        if not os.path.isdir(output_dir):
            os.makedirs(f'{output_dir}/images')
            os.makedirs(f'{output_dir}/annotations')

    no_annot_data = 1
    print(output_dir)
    for idx, image_file in enumerate(image_list):
        img_file_name = image_file.split('/')[-1].split('.')[0]
        # print(idx, img_file_name)

        image = cv2.imread(image_file)
        height, width, _ = image.shape
        cv2.imwrite(f'{output_dir}/images/{idx}.jpg', image)

        annot_file = image_file.split('/')[:-1]
        # print(annot_file)
        # annot_file.insert(-1, 'annotations')
        annot_file[-1] = "annotations"
        annot_file = '/'.join(annot_file)
        annot_file = f'{annot_file}/{img_file_name}.xml'
        print(annot_file)
        # break

        annotation_data = read_annot_file(annot_file)
        write_new_xml(annotation_data, output_dir, idx, width, height)
        # break


def get_file_list(path):
    ds = pathlib.Path(path)

    images = list(ds.glob('*.jpg'))
    images = [str(path) for path in images]
    
    # annot_path = path.split('/')
    # annot_path.insert(5, 'annotations')
    # annot_path = '/'.join(annot_path)
    
    # ds = pathlib.Path(annot_path)
    # annotations = list(ds.glob('*.xml'))
    # annotations = [str(path) for path in annotations]

    # print("image files : ", len(images), "\nannotation files : ", len(annotations))
    # return images, annotations

    return images


if __name__ == "__main__":
    CLASSES = set()
    output_path = "/data/Datasets/Seeds/DMC/set1"
    dataset_path = "/data/Datasets/Seeds/DMC/images"

    total_images = get_file_list(dataset_path)
    train_images, test_images = train_test_split(total_images, test_size=0.2, shuffle=True)

    # train_images = get_file_list(f'{dataset_path}/images/train')
    # valid_images = get_file_list(f'{dataset_path}/images/valid')
    process(train_images, True)
    process(test_images, False)

    label_path = '/'.join(output_path.split('/')[:-1])
    if not os.path.isdir(f"{label_path}/labels"):
        os.makedirs(f"{label_path}/labels")

    f = open(f'{label_path}/labels/labels.txt', 'w')
    for label in sorted(list(CLASSES)):
        f.write(f'{label}\n')

    f.close()