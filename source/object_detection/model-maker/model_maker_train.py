import os
import pandas as pd
import tensorflow as tf
import xml.etree.ElementTree as ET

from tqdm import tqdm
from glob import glob
from tflite_model_maker import object_detector
from tflite_model_maker.config import ExportFormat


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 1:
    try:
        print("Activate Multi GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
    except RuntimeError as e:
        print(e)

else:
    try:
        print("Activate Sigle GPU")
        tf.config.experimental.set_memory_growth(gpus[0], True)
        strategy = tf.distribute.experimental.CentralStorageStrategy()
    except RuntimeError as e:
        print(e)


def load_annot_data(annot_file, target_classes):
    target = ET.parse(annot_file).getroot()

    height = int(target.find('size').find('height').text)
    width = int(target.find('size').find('width').text)

    bboxes, labels = [], []
    for obj in target.iter("object"):
        label = obj.find("name").text.strip()
        if label in target_classes:
            labels.append([label])

            bndbox = obj.find("bndbox")
            bbox = []
            for current in ["xmin", "ymin", "xmax", "ymax"]:
                coordinate = int(float(bndbox.find(current).text))
                if current == "xmin" and coordinate < 0:
                    coordinate = 0
                elif current == "ymin" and coordinate < 0:
                    coordinate = 0
                elif current == "xmax" and coordinate > width:
                    coordinate = width
                elif current == "ymax" and coordinate > height:
                    coordinate = height
                bbox.append(coordinate)
            bboxes.append(bbox)

    return bboxes, labels


def label_check(dir):
    files = sorted(glob(f"{dir}/Annotations/*.xml"))
    targets = set()
    for idx in tqdm(range(len(files))):
        file = files[idx]
        bboxes, labels = load_annot_data(file, CLASSES)
        for label in labels:
            targets.add(label[0])
            if not label[0] in CLASSES:
                print(file, label)
                return False

    print(targets)
    return True

if __name__ == "__main__":
    ROOT_DIR = "/home/ubuntu/Datasets/BR"
    TRAIN_DIR = f"{ROOT_DIR}/set0_384/train"
    VALID_DIR = f"{ROOT_DIR}/set0_384/valid"

    LABEL_FILE = f"{ROOT_DIR}/Labels/labels.txt"
    LABEL_FILE = pd.read_csv(LABEL_FILE, sep=',', index_col=False, header=None)
    CLASSES = LABEL_FILE[0].tolist()
    print(CLASSES)

    train_check = label_check(TRAIN_DIR)
    valid_check = label_check(VALID_DIR)
    
    if train_check and valid_check:
        EPOCHS = 100
        BATCH_SIZE = 64
        MAX_DETECTIONS = 10
        HPARAMS = {"optimizer" : "sgd",
                   "momentum" : 0.9, ## default : 0.9
                   "lr_decay_method" : "cosine",
                   "learning_rate" : 0.008,
                   "lr_warmup_init" : 0.0008,
                   "lr_warmup_epoch" : 1.0, ## default : 1.0
                   "aspect_ratios" : [8.69, 3.89, 1.52, 0.41], ## [8.0, 4.0, 2.0, 1.0, 0.5], [9.44, 4.73, 2.32, 0.96, 0.22]
                   "alpha" : 0.25,
                   "gamma" : 2,
                   "first_lr_drop_epoch" : EPOCHS * (2/3),}

        SAVE_PATH = "/home/ubuntu/Models/efficientdet_lite"
        PROJECT = ROOT_DIR.split('/')[-1]
        DS_NAME = TRAIN_DIR.split('/')[-2]
        MODEL_FILE = f"{PROJECT}-{DS_NAME}-{EPOCHS}"

        train_data = object_detector.DataLoader.from_pascal_voc(images_dir=f"{TRAIN_DIR}/JPEGImages",
                                                                annotations_dir=f"{TRAIN_DIR}/Annotations", 
                                                                label_map=CLASSES)

        validation_data = object_detector.DataLoader.from_pascal_voc(images_dir=f'{VALID_DIR}/JPEGImages',
                                                                     annotations_dir=f'{VALID_DIR}/Annotations',
                                                                     label_map=CLASSES)

        spec = object_detector.EfficientDetLite1Spec(verbose=1,
                                                     strategy="gpus", # 'gpus', None
                                                     hparams=HPARAMS,
                                                     tflite_max_detections=MAX_DETECTIONS,
                                                     model_dir=f'{SAVE_PATH}/{MODEL_FILE}')

        model = object_detector.create(train_data,
                                       model_spec=spec,
                                       epochs=EPOCHS,
                                       batch_size=BATCH_SIZE,
                                       validation_data=validation_data,
                                       do_train=True,
                                       train_whole_model=True,)

        model.export(export_dir=f"{SAVE_PATH}/{MODEL_FILE}",
                     tflite_filename=f'{MODEL_FILE}.tflite',
                     saved_model_filename=f'{SAVE_PATH}/{MODEL_FILE}/saved_model',
                     export_format=[ExportFormat.TFLITE, ExportFormat.SAVED_MODEL])
        print("exported")