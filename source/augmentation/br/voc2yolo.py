import os
import cv2
from glob import glob
from tqdm import tqdm
from shutil import copytree
from utils import read_label_file, read_xml

def yolo2voc(class_id, width, height, x, y, w, h):
    xmin = int((x*width) - (w * width)/2.0)
    ymin = int((y*height) - (h * height)/2.0)
    xmax = int((x*width) + (w * width)/2.0)
    ymax = int((y*height) + (h * height)/2.0)
    class_id = int(class_id)

    return (class_id, xmin, ymin, xmax, ymax)

if __name__ == "__main__":
    FOLDERS = ["train", "valid"]
    ROOT_DIR = "/home/ubuntu/Datasets/BR/set1"
    LABEL_DIR = "/home/ubuntu/Datasets/BR/Labels/labels.txt"

    classes = read_label_file(LABEL_DIR)
    print(classes)

    for folder in FOLDERS:
        images = sorted(glob(f"{ROOT_DIR}/{folder}/images/*"))
        annotations = sorted(glob(f"{ROOT_DIR}/{folder}/annotations/*"))

        if not os.path.isdir(f"{ROOT_DIR}/{folder}/v4set"):
            os.makedirs(f"{ROOT_DIR}/{folder}/v4set")
            os.makedirs(f"{ROOT_DIR}/{folder}/draw")

        print(f"{ROOT_DIR}/{folder}")
        for idx in tqdm(range(len(annotations))):
            annot = annotations[idx]
            file_name = annot.split('/')[-1].split('.')[0]
            image = cv2.imread(f"{ROOT_DIR}/{folder}/images/{file_name}.jpg")
            cv2.imwrite(f"{ROOT_DIR}/{folder}/v4set/{file_name}.jpg", image)

            with open(f"{ROOT_DIR}/{folder}/v4set/{file_name}.txt", "w") as f:
                bboxes, labels = read_xml(annot, classes, format="yolo")
                for bbox, label in zip(bboxes, labels):
                    f.write(str(label) + " " + " ".join([("%.6f" % a) for a in bbox]) + '\n')
                f.close()

        v4_images = glob(f"{ROOT_DIR}/{folder}/v4set/*.jpg")
        with open(f"{ROOT_DIR}/{folder}/files.txt", "w") as f:
            for image in v4_images:
                data = image + '\n'
                f.write(data)
        f.close()