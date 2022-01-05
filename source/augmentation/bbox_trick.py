import cv2
import os
from src.utils import read_xml, make_save_dir, write_xml, read_label_file, get_files, visualize

if __name__ == "__main__":
    ROOT_DIR = "/data/Datasets/SPC"
    FOLDER = "full-name5/valid"
    IMG_DIR = f"{ROOT_DIR}/{FOLDER}/images"
    ANNOT_DIR = f"{ROOT_DIR}/{FOLDER}/annotations"
    LABEL_DIR = f"{ROOT_DIR}/Labels/labels.txt"
    SAVE_DIR = f"{ROOT_DIR}/{FOLDER}/annotations-trick"
    limit_ratio = 4

    if not os.path.isdir(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    classes = read_label_file(LABEL_DIR)
    images, annotations = get_files(IMG_DIR), get_files(ANNOT_DIR)
    for idx, (image, annot) in enumerate(zip(images, annotations)):
        filename = image.split('/')[-1].split('.')[0]
        image = cv2.imread(image)
        height, width = image.shape[:-1]
        bboxes, labels = read_xml(annot, classes, format='pascal_voc')

        t_bboxes = []
        for xmin, ymin, xmax, ymax in bboxes:
            if (xmax - xmin) / (ymax - ymin) > limit_ratio:
                print(f'({xmin}, {ymin}, {xmax}, {ymax}) -> ', end='')
                
                ymed = (ymax + ymin) // 2
                yhlf = (xmax - xmin) // (2 * limit_ratio)
                ymin = max(0, ymed - yhlf)
                ymax = min(height - 1, ymed + yhlf)

                print(f'({xmin}, {ymin}, {xmax}, {ymax})')
                t_bboxes.append((xmin, ymin, xmax, ymax))

        write_xml(SAVE_DIR, t_bboxes, labels, filename, height, width, format='pascal_voc')
        # visualize(image, t_bboxes, labels)    