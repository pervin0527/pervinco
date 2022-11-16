import os
import pandas as pd
import tensorflow as tf
import xml.etree.ElementTree as ET

from tqdm import tqdm
from glob import glob
from tflite_model_maker import object_detector
from tflite_model_maker.config import ExportFormat
from tensorflow_examples.lite.model_maker.third_party.efficientdet.keras import train, train_lib

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

    return True, len(files)


if __name__ == "__main__":
    ROOT_DIR = "/home/ubuntu/Datasets/BR"
    TRAIN_DIR = f"{ROOT_DIR}/seed1_384/set0/train"
    VALID_DIR = f"{ROOT_DIR}/seed1_384/set0/valid"
    CKPT_PATH = "/home/ubuntu/Models/efficientdet_lite/BR-set1-300-final/"

    LABEL_FILE = f"{ROOT_DIR}/Labels/labels.txt"
    LABEL_FILE = pd.read_csv(LABEL_FILE, sep=',', index_col=False, header=None)
    CLASSES = LABEL_FILE[0].tolist()
    print(CLASSES)

    train_check, train_files = label_check(TRAIN_DIR)
    valid_check, valid_files = label_check(VALID_DIR)
    
    if train_check and valid_check:
        EPOCHS = 300
        BATCH_SIZE = 32 * len(gpus)
        MAX_DETECTIONS = 10
        HPARAMS = {
            "optimizer" : "sgd",
            "momentum" : 0.9,
            "lr_decay_method" : "cosine",
            "learning_rate" : 0.008,
            "lr_warmup_init" : 0.0008,
            "lr_warmup_epoch" : 1.0,
            "anchor_scale" : 7.0,
            "aspect_ratios" : [8.0, 4.0, 2.0, 1.0, 0.5],
            "num_scales" : 5,
            "alpha" : 0.25,
            "gamma" : 2,
            "first_lr_drop_epoch" : 200, ## EPOCHS * (2/3)
            "second_lr_drop_epoch" : 250, ## EPOCHS * (6/5)
            "moving_average_decay" : 0.9998
        }

        train_data = object_detector.DataLoader.from_pascal_voc(images_dir=f"{TRAIN_DIR}/JPEGImages",
                                                                annotations_dir=f"{TRAIN_DIR}/Annotations", 
                                                                label_map=CLASSES)

        validation_data = object_detector.DataLoader.from_pascal_voc(images_dir=f'{VALID_DIR}/JPEGImages',
                                                                     annotations_dir=f'{VALID_DIR}/Annotations',
                                                                     label_map=CLASSES)

        spec = object_detector.EfficientDetLite1Spec(verbose=1,
                                                     strategy=None, # 'gpus', None
                                                     hparams=HPARAMS,
                                                     tflite_max_detections=MAX_DETECTIONS,
                                                     model_dir=f'{CKPT_PATH}/export')

        detector = object_detector.create(train_data,
                                          model_spec=spec,
                                          epochs=EPOCHS,
                                          batch_size=BATCH_SIZE,
                                          validation_data=validation_data,
                                          do_train=False,
                                          train_whole_model=True)

        train_ds, steps_per_epoch, _ = detector._get_dataset_and_steps(train_data, BATCH_SIZE, is_training=True)
        validation_ds, validation_steps, val_json_file = detector._get_dataset_and_steps(validation_data, BATCH_SIZE, is_training=False)

        with strategy.scope():
            model = detector.create_model()
            config = spec.config
            config.update(
                dict(
                    steps_per_epoch=steps_per_epoch,
                    eval_samples=BATCH_SIZE*validation_steps,
                    val_json_file=val_json_file,
                    batch_size=BATCH_SIZE
                )
            )

            train.setup_model(model, config)

            if CKPT_PATH:
                latest = tf.train.latest_checkpoint(CKPT_PATH)
                model.load_weights(latest)

            model.summary()

        detector.model = model
        detector.export(export_dir=f"{CKPT_PATH}/export",
                        tflite_filename='model.tflite',
                        saved_model_filename=f'{CKPT_PATH}/export/saved_model',
                        export_format=[ExportFormat.TFLITE, ExportFormat.SAVED_MODEL])
        print("exported")