import cv2
import random
import albumentations as A
from glob import glob
from src.utils import read_label_file, read_xml, get_files, visualize

def calculate_bbox(bboxes, labels):
    for xmin, ymin, xmax, ymax in bboxes:
        bbox_width = xmax - xmin
        bbox_height = ymax - ymin

        bbox_ratio = int(bbox_width / bbox_height)
        if bbox_ratio > max_ratio:
            max_ratio = bbox_ratio
            area = int(bbox_width * bbox_height)
            print(max_ratio, area)
            visualize(image, bboxes, labels)

if __name__ == "__main__":
    IMG_SIZE = 384
    ROOT_PATH = "/data/Datasets/SPC-Hannam"
    LABEL_PATH = f"{ROOT_PATH}/Labels/labels.txt"
    classes = read_label_file(LABEL_PATH)

    # IMAGE_PATH = f"{ROOT_PATH}/full-name12/images"
    # ANNOT_PATH = f"{ROOT_PATH}/full-name12/annotations"
    # images, annotations = get_files(IMAGE_PATH), get_files(ANNOT_PATH)
    
    IMAGE_PATH = f"{ROOT_PATH}/ver4/images"
    ANNOT_PATH = f"{ROOT_PATH}/ver4/annotations"
    images = sorted(glob(f"{IMAGE_PATH}/*"))
    annotations = sorted(glob(f"{ANNOT_PATH}/*"))
    print(len(images), len(annotations))

    dataset = list(zip(images, annotations))
    random.shuffle(dataset)

    max_ratio = 0
    for (image, annot) in dataset:
        # if annot == "/data/Datasets/SPC/full-name11/train/annotations/train_0_3652.xml":
        image = cv2.imread(image)
        bboxes, labels = read_xml(annot, classes, format='pascal_voc')

        transform = A.Compose([
            A.Resize(IMG_SIZE, IMG_SIZE, p=1),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

        transformed = transform(image=image, bboxes=bboxes, labels=labels)
        image, bboxes, labels = transformed['image'], transformed['bboxes'], transformed['labels']
        
        if bboxes:
            visualize(image, bboxes, labels)