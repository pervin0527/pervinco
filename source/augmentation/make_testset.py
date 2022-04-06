from cProfile import label
import cv2
import random
import albumentations as A
from glob import glob
from tqdm import tqdm
from src.custom_aug import mixup
from src.utils import read_label_file, read_xml, visualize, get_files, make_save_dir, write_xml

def create_testset():
    transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE, p=1),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
    dataset = list(zip(images, annotations))

    for index in tqdm(range(ITER)):
        data = random.sample(dataset, 1)[0]
        image, annot = data
        dataset.pop(dataset.index(data))

        image = cv2.imread(image)
        bboxes, labels = read_xml(annot, classes, format='pascal_voc')

        if IS_NORMAL:
            transformed = transform(image=image, bboxes=bboxes, labels=labels)
            image, bboxes, labels = transformed['image'], transformed['bboxes'], transformed['labels']

        else:
            image, bboxes, labels = mixup(image, bboxes, labels, IMG_SIZE, MX_BG, min=0.4, max=0.45, alpha=1.0)

        cv2.imwrite(f"{SAVE_DIR}/images/{index:>04}.jpg", image)
        write_xml(f"{SAVE_DIR}/annotations", bboxes, labels, f"{index:>04}", image.shape[0], image.shape[1], 'pascal_voc')


if __name__ == "__main__":
    ROOT_DIR = "/data/Datasets/SPC"
    FOLDER = "sample-set1"

    IMG_DIR = f"{ROOT_DIR}/{FOLDER}/images"
    ANNOT_DIR = f"{ROOT_DIR}/{FOLDER}/annotations"
    LABEL_DIR = f"{ROOT_DIR}/Labels/labels.txt"
    SAVE_DIR = f"{ROOT_DIR}/{FOLDER}/test"

    IS_NORMAL = True
    ITER = 100
    IMG_SIZE = 512
    MX_BG = "/data/Datasets/Mixup_background"

    classes = read_label_file(LABEL_DIR)
    images, annotations = get_files(IMG_DIR), get_files(ANNOT_DIR)

    make_save_dir(SAVE_DIR)
    create_testset()
