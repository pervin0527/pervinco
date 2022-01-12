import os
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
    TXT_DIR = "/data/Datasets/SPC/set1/train/augmentations/labels"
    LABEL_DIR = "/data/Datasets/SPC/Labels/labels.txt"
    SAVE_DIR = f"{('/').join(TXT_DIR.split('/')[:-1])}/labels"

    classes = read_label_file(LABEL_DIR)
    txts = get_files(TXT_DIR)
    print(len(txts))

    for txt in txts:
        filename = txt.split('/')[-1].split('.')[0]

        df = pd.read_csv(txt, sep=',', header=None, index_col=False)
        df_list = df[0].tolist()

        labels, bboxes = get_annot_data(df_list)
        print(labels, bboxes)

        # for label, bbox in zip(labels, bboxes):
        #     result = unconvert(1440, 1440, bbox[0], bbox[1], bbox[2], bbox[3])
        #     print(result)
        write_xml(SAVE_DIR, bboxes, labels, filename, 1440, 1440, format="yolo")
        break