import os
import cv2
import pandas as pd
from src.utils import read_label_file, write_xml, get_files

def get_annot_data(txt_df):
    bboxes, labels = [], []
    for data in txt_df:
        data = data.split(' ')
        label, xmin, ymin, xmax, ymax = classes[int(data[0])], float(data[1]), float(data[2]), float(data[3]), float(data[4])
        labels.append(label)
        bboxes.append((xmin, ymin, xmax, ymax))

    return labels, bboxes

def unconvert(width, height, x, y, w, h):
    xmax = int((x*width) + (w * width)/2.0)
    xmin = int((x*width) - (w * width)/2.0)
    ymax = int((y*height) + (h * height)/2.0)
    ymin = int((y*height) - (h * height)/2.0)
    
    return (xmin, xmax, ymin, ymax)

if __name__ == "__main__":
    TXT_DIR = "/home/barcelona/yolov5-tflite/yolov5-5.0/runs/detect/exp3/labels"
    LABEL_DIR = "/data/Datasets/SPC/Labels/labels.txt"
    IMG_DIR = "/data/Datasets/SPC/Testset/day_night2/original/images"
    SAVE_DIR = f"{('/').join(TXT_DIR.split('/')[:-1])}/txt2xml"
    IMG_SIZE = 384

    classes = read_label_file(LABEL_DIR)
    txts = get_files(TXT_DIR)
    print(len(txts))

    images = get_files(IMG_DIR)
    print(len(images))

    if not os.path.isdir(SAVE_DIR):
        os.makedirs(f"{SAVE_DIR}/annotations")
        os.makedirs(f"{SAVE_DIR}/images")

    for txt in txts:
        filename = txt.split('/')[-1].split('.')[0]
        image = cv2.imread(f"{IMG_DIR}/{filename}.jpg")
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        height, width = image.shape[:-1]

        df = pd.read_csv(txt, sep=',', header=None, index_col=False)
        df_list = df[0].tolist()

        labels, bboxes = get_annot_data(df_list)
        write_xml(f"{SAVE_DIR}/annotations", bboxes, labels, filename, height, width, format="yolo")
        cv2.imwrite(f"{SAVE_DIR}/images/{filename}.jpg", image)