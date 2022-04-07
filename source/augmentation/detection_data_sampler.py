import os
import cv2
import random
import albumentations as A
from glob import glob
from tqdm import tqdm
from src.utils import read_label_file, read_xml, write_xml, make_save_dir

def expand_bbox(bboxes, height):
    expand = []
    for xmin, ymin, xmax, ymax in bboxes:
        if (xmax - xmin) / (ymax - ymin) > limit_ratio:
            if (ymin - GAP) > 0 and (ymax + GAP) < (height - 1):
                ymin -= GAP
                ymax += GAP
                expand.append((xmin, ymin, xmax, ymax))

    return expand

if __name__ == "__main__":
    ROOT_DIR = "/data/Datasets/SPC"
    LABEL_DIR = f"{ROOT_DIR}/Labels/labels.txt"
    FOLDER = "Cvat"
    GAP = 8
    IMG_SIZE = 320
    SAVE_DIR = f"{ROOT_DIR}/sample-set2"

    number = 300
    limit_ratio = 10
    finish_set = set()
    classes = read_label_file(LABEL_DIR)

    transform = A.Compose([A.Resize(IMG_SIZE, IMG_SIZE, p=1)
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

    make_save_dir(SAVE_DIR)
    for label in classes:
        print(label)
        folders = sorted(glob(f"{ROOT_DIR}/{FOLDER}/{label}/*"))
        
        for index, folder in enumerate(folders, start=1):
            images = sorted(glob(f"{folder}/JPEGImages/*"))
            annotations = sorted(glob(f"{folder}/Annotations/*"))
            dataset = list(zip(images, annotations))

            for idx in tqdm(range(number)):
                try:
                    random.shuffle(dataset)
                    data = random.sample(dataset, 1)
                    image, annot = data[0][0], data[0][1]

                    image = cv2.imread(image)
                    bboxes, labels = read_xml(annot, classes, format='pascal_voc')

                    for label in labels:
                        finish_set.add(label)

                    resized =transform(image=image, bboxes=bboxes, labels=labels)
                    resized_image, resized_bboxes, resized_labels = resized['image'], resized['bboxes'], resized['labels']

                    if "Paris_baguette" in labels:
                        expanded = expand_bbox(resized_bboxes, IMG_SIZE)
                        if expanded:
                            resized_bboxes = expanded[:]

                    cv2.imwrite(f"{SAVE_DIR}/images/{label}_{index}_{idx:>05}.jpg", resized_image)
                    write_xml(f"{SAVE_DIR}/annotations", resized_bboxes, resized_labels, f"{label}_{index}_{idx:>05}", IMG_SIZE, IMG_SIZE, format='pascal_voc')

                except:
                    pass

    print(finish_set)