import os
import pandas as pd
import tensorflow as tf

from tflite_model_maker import object_detector
from tflite_model_maker.config import ExportFormat, QuantizationConfig


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
    ROOT_DIR = "/home/ubuntu/Datasets/BR"
    TRAIN_DIR = f"{ROOT_DIR}/set1_384/train"
    VALID_DIR = f"{ROOT_DIR}/set1_384/valid"

    LABEL_FILE = f"{ROOT_DIR}/Labels/labels.txt"
    LABEL_FILE = pd.read_csv(LABEL_FILE, sep=',', index_col=False, header=None)
    CLASSES = LABEL_FILE[0].tolist()
    print(CLASSES)
    
    EPOCHS = 10
    BATCH_SIZE = 64
    MAX_DETECTIONS = 10

    HPARAMS = {"optimizer" : "sgd",
               "momentum" : 0.9, ## default : 0.9
               "lr_decay_method" : "cosine",
               "learning_rate" : 0.008,
               "lr_warmup_init" : 0.0008,
               
               "lr_warmup_epoch" : 1.0, ## default : 1.0
               "anchor_scale" : 3.0, ## 4.0
               "aspect_ratios" : [8.85, 4.06, 1.67, 0.53], ##[8.0, 4.0, 2.0, 1.0, 0.5],
               "num_scales" : 5,
               "alpha" : 0.25,
               "gamma" : 2,
               "first_lr_drop_epoch" : EPOCHS * (2/3)}

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

    tf.saved_model.save(model, f"{SAVE_PATH}/{MODEL_FILE}/FLOAT32/saved_model")

    # config = QuantizationConfig.for_float16()
    # model.export(export_dir=f"{SAVE_PATH}/{MODEL_FILE}/FLOAT16",
    #              quantization_config=config,
    #              saved_model_filename=f"{SAVE_PATH}/{MODEL_FILE}/FLOAT16/saved_model",
    #              export_format=[ExportFormat.SAVED_MODEL])                   

    model.export(export_dir=f"{SAVE_PATH}/{MODEL_FILE}",
                 label_filename=f'{SAVE_PATH}/label_map.txt',
                 tflite_filename=f'{MODEL_FILE}.tflite',
                 saved_model_filename=f'{SAVE_PATH}/{MODEL_FILE}/saved_model',
                 export_format=[ExportFormat.TFLITE, ExportFormat.SAVED_MODEL])