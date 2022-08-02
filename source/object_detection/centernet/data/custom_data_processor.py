import os
import xml.etree.ElementTree as ET

def convert_voc_annotation(data_path, folder, txt_file, annot_path, use_difficul_bbox=True):
    data_list_file = f"{data_path}/{folder}/{txt_file}"
    annot_path = f"{data_path}/{folder}/annot.txt"

    with open(data_list_file, "r") as f:
        lines = f.readlines()
        files = [line.strip() for line in lines]

    with open(annot_path, 'w') as f:
        for file in files:
            img_path = f"{data_path}/{folder}/images/{file}.jpg"
            xml_path = f"{data_path}/{folder}/annotations/{file}.xml"

            annotation = img_path

            root = ET.parse(xml_path).getroot()
            objects = root.findall("object")
            for obj in objects:
                difficult = obj.find("difficult").text.strip()

                if (not use_difficul_bbox) and (int(difficult) == 1):
                    continue

                bbox = obj.find("bndbox")
                class_ind = classes.index(obj.find("name").text.lower().strip())
                xmin = bbox.find('xmin').text.strip()
                xmax = bbox.find('xmax').text.strip()
                ymin = bbox.find('ymin').text.strip()
                ymax = bbox.find('ymax').text.strip()
                annotation += ' ' + ','.join([xmin, ymin, xmax, ymax, str(class_ind)])

            print(annotation)
            f.write(annotation + "\n")

    return len(file)


if __name__ == "__main__":
    classes = ["face"]
    data_path = "/home/ubuntu/Datasets/WIDER/FACE"

    train_data = convert_voc_annotation(data_path, "train_512", "list.txt", False)
    test_data = convert_voc_annotation(data_path, "test_512", "list.txt", False)