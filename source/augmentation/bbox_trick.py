import cv2
import os
import albumentations as A
from src.utils import read_xml, make_save_dir, write_xml, read_label_file, get_files, visualize

if __name__ == "__main__":
    ROOT_DIR = "/data/Datasets/SPC"
    FOLDER = "pb"
    IMG_DIR = f"{ROOT_DIR}/{FOLDER}/images"
    ANNOT_DIR = f"{ROOT_DIR}/{FOLDER}/annotations"
    LABEL_DIR = f"{ROOT_DIR}/Labels/labels.txt"
    SAVE_DIR = f"{ROOT_DIR}/{FOLDER}/annotations-trick"
    limit_ratio = 10
    visual = False

    GAP = 3
    IMG_SIZE = 384
    transform = A.Compose([A.Resize(IMG_SIZE, IMG_SIZE, p=1),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

    if not os.path.isdir(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    classes = read_label_file(LABEL_DIR)
    images, annotations = get_files(IMG_DIR), get_files(ANNOT_DIR)
    for idx, (image, annot) in enumerate(zip(images, annotations)):
        filename = image.split('/')[-1].split('.')[0]
        image = cv2.imread(image)
        height, width = image.shape[:-1]
        bboxes, labels = read_xml(annot, classes, format='pascal_voc')

        transformed = transform(image=image, bboxes=bboxes, labels=labels)
        t_image, t_bboxes, t_labels = transformed['image'], transformed['bboxes'], transformed['labels']

        confirm = []
        for xmin, ymin, xmax, ymax in t_bboxes:
            if (xmax - xmin) / (ymax - ymin) > limit_ratio:               
                # ymed = (ymax + ymin) // 2
                # yhlf = (xmax - xmin) // (2 * limit_ratio)
                # ymin = max(0, ymed - yhlf)
                # ymax = min(height - 1, ymed + yhlf)
                if ymin - GAP > 0 and ymax < height - 1:
                    ymin -= GAP
                    ymax += GAP

                else:
                    ymin = 0
                    ymax = height - 1
                
            confirm.append((xmin, ymin, xmax, ymax))

        write_xml(SAVE_DIR, confirm, labels, filename, height, width, format='pascal_voc')
        
        if confirm and visual:
            print(filename)
            visualize(t_image, confirm, labels)    