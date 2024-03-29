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
    SAVE_PATH = "/home/ubuntu/Models/BR_FINAL"
    TRAIN_DIR = f"{ROOT_DIR}/seed1_384/fold00/train"
    VALID_DIR = f"{ROOT_DIR}/seed1_384/fold00/valid"
    CKPT_DIR = "/home/ubuntu/Models/efficientdet_lite/BR-set4-200"
    
    LABEL_FILE = f"{ROOT_DIR}/Labels/labels.txt"
    LABEL_FILE = pd.read_csv(LABEL_FILE, sep=',', index_col=False, header=None)
    CLASSES = LABEL_FILE[0].tolist()
    print(CLASSES)
    
    FOLDS = len(glob(f"{TRAIN_DIR}/*"))
    EPOCHS = 100
    BATCH_SIZE = 64 * len(gpus)
    MAX_DETECTIONS = 10
    HPARAMS = {
        "optimizer" : "sgd",
        "momentum" : 0.9,
        "lr_decay_method" : "cosine",
        "learning_rate" : 0.008,
        "lr_warmup_init" : 0.0008,
        "lr_warmup_epoch" : 1.0,
        "aspect_ratios" : [8.24, 4.42, 2.2, 0.92],
        "anchor_scale" : 5,
        "num_scales" : 4,
        "alpha" : 0.25,
        "gamma" : 2,
        "max_instances_per_image" : 10
    }

    PROJECT = ROOT_DIR.split('/')[-1]
    DS_NAME = TRAIN_DIR.split('/')[-2]
    MODEL_FILE = f"{PROJECT}-{DS_NAME}-f{FOLDS}"

    for f in range(FOLDS):
        train_data = object_detector.DataLoader.from_pascal_voc(images_dir=f"{TRAIN_DIR}/{f}/JPEGImages",
                                                                annotations_dir=f"{TRAIN_DIR}/{f}/Annotations", 
                                                                label_map=CLASSES)

        validation_data = object_detector.DataLoader.from_pascal_voc(images_dir=f'{VALID_DIR}/JPEGImages',
                                                                    annotations_dir=f'{VALID_DIR}/Annotations',
                                                                    label_map=CLASSES)

        spec = object_detector.EfficientDetLite1Spec(verbose=1,
                                                     strategy="gpus", # 'gpus', None
                                                     hparams=HPARAMS,
                                                     tflite_max_detections=MAX_DETECTIONS,
                                                     model_dir=f'{SAVE_PATH}/{MODEL_FILE}')

        detector = object_detector.create(train_data,
                                          model_spec=spec,
                                          epochs=EPOCHS,
                                          batch_size=BATCH_SIZE,
                                          validation_data=validation_data,
                                          do_train=False,
                                          train_whole_model=True)

        train_ds, steps_per_epoch, _ = detector._get_dataset_and_steps(train_data, BATCH_SIZE, is_training=True)
        validation_ds, validation_steps, val_json_file = detector._get_dataset_and_steps(validation_data, BATCH_SIZE, is_training=False)

        config = spec.config
        config.update(dict(steps_per_epoch=steps_per_epoch,
                           eval_samples=BATCH_SIZE*validation_steps,
                           val_json_file=val_json_file,
                           batch_size=BATCH_SIZE))
        if not tf.io.gfile.makedirs(f"{SAVE_PATH}/{MODEL_FILE}"):
            config_file = os.path.join(f"{SAVE_PATH}/{MODEL_FILE}", 'config.yaml')   
            tf.io.gfile.GFile(config_file, 'w').write(str(config))

        os.system("clear")
        print(f"Fold : {f}")
        if f == 0:
            with strategy.scope():
                model = detector.create_model()
                train.setup_model(model, config)

                try:
                    latest = tf.train.latest_checkpoint(f"{CKPT_DIR}/")
                    last_epoch = int(latest.split('/')[-1].split("-")[1])
                    model.load_weights(latest)
                    print("Checkpoint found {}".format(latest))

                except Exception as e:
                    print("Checkpoint not found: ", e)

                model.summary()
                model.fit(train_ds,
                          initial_epoch=0, 
                          epochs=EPOCHS,
                          steps_per_epoch=steps_per_epoch,
                          validation_data=validation_ds,
                          validation_steps=validation_steps,
                          callbacks=train_lib.get_callbacks(config.as_dict(), validation_ds))
        else:
            with strategy.scope():
                model = detector.create_model()
                train.setup_model(model, config)

                try:
                    latest = tf.train.latest_checkpoint(f"{SAVE_PATH}/{MODEL_FILE}/")
                    last_epoch = int(latest.split('/')[-1].split("-")[1])
                    model.load_weights(latest)
                    print("Checkpoint found {}".format(latest))

                except Exception as e:
                    print("Checkpoint not found: ", e)

                model.fit(train_ds,
                          initial_epoch=0, 
                          epochs=EPOCHS,
                          steps_per_epoch=steps_per_epoch,
                          validation_data=validation_ds,
                          validation_steps=validation_steps,
                          callbacks=train_lib.get_callbacks(config.as_dict(), validation_ds))

        del train_data, validation_data, spec, detector, train_ds, steps_per_epoch, validation_ds, validation_steps, model

    detector.model = model
    detector.export(export_dir=f"{SAVE_PATH}/{MODEL_FILE}",
                    tflite_filename='model.tflite',
                    saved_model_filename=f'{SAVE_PATH}/{MODEL_FILE}/saved_model',
                    export_format=[ExportFormat.TFLITE, ExportFormat.SAVED_MODEL])
    print("exported")