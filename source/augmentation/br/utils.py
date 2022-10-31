import os
import random
import xml.etree.ElementTree as ET

from glob import glob

def make_file_list(dirs, num_valid):
    total_files = []
    for dir in dirs:
        images = sorted(glob(f"{dir}/*/JPEGImages/*"))
        annotations = sorted(glob(f"{dir}/*/Annotations/*"))
        print(f"{dir} - image_files : {len(images)},  annot_files : {len(annotations)}")

        total_files.extend(list(zip(images, annotations)))

    print(f"total_files : {len(total_files)}")
    random.shuffle(total_files)

    train_files = total_files[:-num_valid]
    valid_files = total_files[-num_valid:]
    print(f"Train files : {len(train_files)}")
    print(f"Valid files : {len(valid_files)}")

    return train_files, valid_files

def make_save_Dir(dir):
    if not os.path.isdir(dir):
        os.makedirs(f"{dir}/JPEGImages")
        os.makedirs(f"{dir}/Annotations")
        os.makedirs(f"{dir}/Results")

def load_annot_data(annot_file):
    target = ET.parse(annot_file).getroot()

    bboxes, labels = [], []
    for obj in target.iter("object"):
        label = obj.find("name").text.strip()
        labels.append([label])

        bndbox = obj.find("bndbox")
        bbox = []
        for current in ["xmin", "ymin", "xmax", "ymax"]:
            coordinate = int(float(bndbox.find(current).text))
            bbox.append(coordinate)
        bboxes.append(bbox)

    return bboxes, labels