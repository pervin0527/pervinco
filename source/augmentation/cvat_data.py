import os
import cv2
import albumentations as A
from glob import glob
from tqdm import tqdm
from src.utils import read_label_file, read_xml, get_files, get_content_filename, write_xml, visualize, make_save_dir

if __name__ == "__main__":
    ROOT_DIR = "/data/Datasets/SPC"
    LABEL_DIR = f"{ROOT_DIR}/Labels/labels.txt"
    FOLDER = "Cvat"
    IMG_SIZE = 384
    SAVE_DIR = f"{ROOT_DIR}/test"

    classes = read_label_file(LABEL_DIR)
    print(classes)

    dataset = sorted(glob(f"{ROOT_DIR}/{FOLDER}/*"))
    dataset = dataset[:-1]
    print(dataset)

    transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE, p=1),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))    

    if not os.path.isdir(f"{SAVE_DIR}"):
        os.makedirs(f"{SAVE_DIR}/images")
        os.makedirs(f"{SAVE_DIR}/annotations")

    for data in dataset:
        name = data.split('/')[-1]
        images =  sorted(glob(f"{data}/*/JPEGImages/*"))
        annotations = sorted(glob(f"{data}/*/Annotations/*"))

        for idx in tqdm(range(len(images))):
            try:
                image = cv2.imread(images[idx])
                bboxes, labels = read_xml(annotations[idx], classes, format="pascal_voc")

                transformed = transform(image=image, bboxes=bboxes, labels=labels)
                transformed_image, transformed_bboxes, transformed_labels = transformed['image'], transformed['bboxes'], transformed['labels']

                cv2.imwrite(f"{SAVE_DIR}/images/{name}_{idx}.jpg", transformed_image)
                write_xml(f"{SAVE_DIR}/annotations", transformed_bboxes, transformed_labels, f"{name}_{idx}", IMG_SIZE, IMG_SIZE, format='pascal_voc')

            except:
                pass
