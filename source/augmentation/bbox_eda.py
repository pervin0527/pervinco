import cv2
import random
import albumentations as A
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

IMG_SIZE = 384
ROOT_PATH = "/data/Datasets/SPC"
LABEL_PATH = f"{ROOT_PATH}/Labels/labels.txt"

IMAGE_PATH = f"{ROOT_PATH}/full-name11/train/images"
ANNOT_PATH = f"{ROOT_PATH}/full-name11/train/annotations"

classes = read_label_file(LABEL_PATH)
images, annotations = get_files(IMAGE_PATH), get_files(ANNOT_PATH)

dataset = list(zip(images, annotations))
random.shuffle(dataset)

max_ratio = 0
for (image, annot) in dataset:
    image = cv2.imread(image)
    bboxes, labels = read_xml(annot, classes, format='pascal_voc')

    transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE, p=1),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

    transformed = transform(image=image, bboxes=bboxes, labels=labels)
    image, bboxes, labels = transformed['image'], transformed['bboxes'], transformed['labels']
    
    if bboxes:
        visualize(image, bboxes, labels)