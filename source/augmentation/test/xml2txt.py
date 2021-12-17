import os
from src.utils import read_label_file, read_xml, make_save_dir, get_files

def yolo2voc(class_id, width, height, x, y, w, h):
    xmin = int((x*width) - (w * width)/2.0)
    ymin = int((y*height) - (h * height)/2.0)
    xmax = int((x*width) + (w * width)/2.0)
    ymax = int((y*height) + (h * height)/2.0)
    class_id = int(class_id)

    return (class_id, xmin, ymin, xmax, ymax)

if __name__ == "__main__":
    ANNOT_DIR = "/data/Datasets/SPC/set1/train/annotations"
    LABEL_DIR = "/data/Datasets/SPC/Labels/labels.txt"
    SAVE_DIR = f"{('/').join(ANNOT_DIR.split('/')[:-1])}/labels"
    print(SAVE_DIR)

    if not os.path.isdir(SAVE_DIR):
        os.mkdir(SAVE_DIR)
    classes = read_label_file(LABEL_DIR)
    print(classes)

    annotations = get_files(ANNOT_DIR)
    print(len(annotations))

    for annot in annotations:
        filename = annot.split('/')[-1].split('.')[0]
        
        with open(f"{SAVE_DIR}/{filename}.txt", 'w') as f:
            bboxes, labels = read_xml(annot, classes, format="yolo")
            # print(bboxes, labels)
            for bbox, label in zip(bboxes, labels):
                f.write(str(label) + " " + " ".join([("%.6f" % a) for a in bbox]) + '\n')