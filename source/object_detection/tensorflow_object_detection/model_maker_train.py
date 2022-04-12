import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
import tensorflow as tf

from absl import logging
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector
from tflite_model_maker.config import ExportFormat
from tflite_model_maker.config import QuantizationConfig

if __name__ == "__main__":
    ROOT_DIR = "/data/Datasets/SPC"
    TRAIN_DIR = f"{ROOT_DIR}/sample-set2/train2"
    VALID_DIR = f"{ROOT_DIR}/sample-set2/valid2"

    LABEL_FILE = f"{ROOT_DIR}/Labels/labels.txt"
    LABEL_FILE = pd.read_csv(LABEL_FILE, sep=',', index_col=False, header=None)
    CLASSES = LABEL_FILE[0].tolist()
    # CLASSES = CLASSES[:-1]
    print(CLASSES)
    
    EPOCHS = 50
    BATCH_SIZE = 64
    MAX_DETECTIONS = 10

    HPARAMS = {"optimizer" : "sgd",
               "learning_rate" : 0.008,
               "lr_warmup_init" : 0.0008,
               "anchor_scale" : 7.0,
               "aspect_ratios" : [8.0, 4.0, 2.0, 1.0, 0.5],
               "num_scales" : 5,
               "alpha" : 0.25,
               "gamma" : 2,
               "es" : False,
               "es_monitor" : "val_det_loss",
               "es_patience" : 15,
               "ckpt" : None}

    # HPARAMS = {"optimizer" : "sgd",
    #            "learning_rate" : 0.08, # 0.008
    #            "lr_warmup_init" : 0.008, # 0.0008
    #            "anchor_scale" : 4.0, # 7.0
    #            "aspect_ratios" : [1.0, 2.0, 0.5], # [8.0, 4.0, 2.0, 1.0, 0.5]
    #            "num_scales" : 3, # 5
    #            "alpha" : 0.25,
    #            "gamma" : 1.5, # 2
    #            "es" : False,
    #            "es_monitor" : "val_det_loss",
    #            "es_patience" : 15,
    #            "ckpt" : None}

    ## TEST
    # HPARAMS = {"optimizer" : "sgd",
    #            "learning_rate" : 0.08,
    #            "lr_warmup_init" : 0.008,
    #            "anchor_scale" : [1.0, 3.0, 5.0, 7.0, 9.0],
    #            "aspect_ratios" : [1.16, 3.48, 9.23, 20.39],
    #            "num_scales" : 3,
    #            "alpha" : 0.25,
    #            "gamma" : 2,
    #            "es" : False,
    #            "es_monitor" : "val_det_loss",
    #            "es_patience" : 15,
    #            "ckpt" : None}

    SAVE_PATH = "/data/Models/efficientdet_lite"
    PROJECT = ROOT_DIR.split('/')[-1]
    DS_NAME = TRAIN_DIR.split('/')[-2]
    MODEL_FILE = f"{PROJECT}-{DS_NAME}-{EPOCHS}"
    # MODEL_FILE = "TEST"

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

    # config = QuantizationConfig.for_int8(representative_data=validation_data)
    # config = QuantizationConfig.for_float16()                                   

    model.export(export_dir=f"{SAVE_PATH}/{MODEL_FILE}",
                 label_filename=f'{SAVE_PATH}/label_map.txt',
                 tflite_filename=f'{MODEL_FILE}.tflite',
                 saved_model_filename=f'{SAVE_PATH}/{MODEL_FILE}/saved_model',
                 export_format=[ExportFormat.TFLITE, ExportFormat.SAVED_MODEL],)
                #  quantization_config=config