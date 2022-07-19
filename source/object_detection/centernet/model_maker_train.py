import os
import pandas as pd
import tensorflow as tf
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

if __name__ == "__main__":
    ROOT_DIR = "/data/Datasets/WFLW"
    TRAIN_DIR = f"{ROOT_DIR}/CUSTOM_XML/augmentation"
    VALID_DIR = f"{ROOT_DIR}/CUSTOM_XML/test"

    LABEL_FILE = f"{ROOT_DIR}/Labels/labels.txt"
    LABEL_FILE = pd.read_csv(LABEL_FILE, sep=',', index_col=False, header=None)
    CLASSES = LABEL_FILE[0].tolist()
    print(CLASSES)
    
    EPOCHS = 300
    BATCH_SIZE = 64
    MAX_DETECTIONS = 10

    HPARAMS = {
        "optimizer" : "sgd",
        "learning_rate" : 0.08, # 0.008
        "lr_warmup_init" : 0.008, # 0.0008
        "anchor_scale" : 4.0, # 7.0
        "aspect_ratios" : [1.0, 2.0, 0.5], # [8.0, 4.0, 2.0, 1.0, 0.5]
        "num_scales" : 3, # 5
        "alpha" : 0.25,
        "gamma" : 1.5, # 2
    }

    SAVE_PATH = "/data/Models/face_detection"
    PROJECT = ROOT_DIR.split('/')[-1]
    DS_NAME = TRAIN_DIR.split('/')[-2]
    MODEL_FILE = f"{PROJECT}-{DS_NAME}-{EPOCHS}"

    train_data = object_detector.DataLoader.from_pascal_voc(images_dir=f"{TRAIN_DIR}/images",
                                                            annotations_dir=f"{TRAIN_DIR}/annotations", 
                                                            label_map=CLASSES)

    validation_data = object_detector.DataLoader.from_pascal_voc(images_dir=f'{VALID_DIR}/images',
                                                                 annotations_dir=f'{VALID_DIR}/annotations',
                                                                 label_map=CLASSES)

    spec = object_detector.EfficientDetLite0Spec(verbose=1,
                                                 strategy=None, # 'gpus'
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