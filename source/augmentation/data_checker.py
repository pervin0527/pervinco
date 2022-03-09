from distutils import extension
import os
import pandas as pd
from glob import glob
from tqdm import tqdm

def read_label_file(path):
    df = pd.read_csv(path, sep=',', index_col=False, header=None)
    labels = df[0].tolist()

    return labels

ROOT = "/data/Datasets/SPC"
LABEL_FILE_PATH = f"{ROOT}/Labels/labels.txt"
CLASSES = read_label_file(LABEL_FILE_PATH)
print(CLASSES)

for c in CLASSES:
    folders = sorted(glob(f"{ROOT}/Cvat/{c}/*"))

    for folder in folders:
        images_path, annotations_path = f"{folder}/JPEGImages", f"{folder}/Annotations"
        images = sorted(glob(f"{images_path}/*"))
        annotations = sorted(glob(f"{annotations_path}/*"))

        if len(images) == len(annotations):
            # print(f"{folder} files SAME, images : {len(images)}, annotations : {len(annotations)}")
            pass

        else:
            print(images_path, annotations_path)

            extension = images[0].split('/')[-1].split('.')[-1]
            for xml_file in annotations:
                file_name = xml_file.split('/')[-1].split('.')[0]

                if not f"{images_path}/{file_name}.{extension}" in images:
                    print(xml_file)
                    os.remove(xml_file)
