import cv2
import os
import albumentations as A
from glob import glob
from src.utils import read_xml, make_save_dir, write_xml, read_label_file, get_files, visualize

if __name__ == "__main__":
    ROOT_DIR = "/data/Datasets/SPC"
    FOLDER = "Cvat/Paris_baguette"
    SAVE_DIR = f"{ROOT_DIR}/full-name-test"
    LABEL_DIR = f"{ROOT_DIR}/Labels/labels.txt"
    classes = read_label_file(LABEL_DIR)

    GAP = 6
    limit_ratio = 10
    visual = False
    IMG_SIZE = 384
    transform = A.Compose([A.Resize(IMG_SIZE, IMG_SIZE, p=1),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

    if not os.path.isdir(f"{SAVE_DIR}/images") and not os.path.isdir(f"{SAVE_DIR}/annotations"):
        os.makedirs(f"{SAVE_DIR}/images")
        os.makedirs(f"{SAVE_DIR}/annotations")

    idx = 0
    for f in sorted(glob(f"{ROOT_DIR}/{FOLDER}/*")):
        images = get_files(f"{f}/JPEGImages")
        annotations = get_files(f"{f}/Annotations")

        for (image, annot) in zip(images, annotations):
            print(annot)
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

            cv2.imwrite(f"{SAVE_DIR}/images/trick_{idx}_{filename}.jpg", t_image)
            write_xml(f"{SAVE_DIR}/annotations", confirm, labels, f"trick_{idx}_{filename}", height, width, format='pascal_voc')
            idx += 1
            
            if confirm and visual:
                print(filename)
                visualize(t_image, confirm, labels)    