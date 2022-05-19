import os
import yaml
import numpy as np
import pandas as pd

def read_label_file(path):
    label_df = pd.read_csv(path, lineterminator="\n", header=None, index_col=False)
    CLASSES = label_df[0].to_list()

    return CLASSES

def send_params(show_contents):
    param.update({"CLASSES":read_label_file(param["LABEL_PATH"])})
    if show_contents:
        print(param)

    return param
    
def save_params():
    if not os.path.isdir(param["SAVE_PATH"]):
        os.makedirs(param["SAVE_PATH"])

    del param["COLORMAP"]
    with open(param["SAVE_PATH"] + "/config.yaml", "w") as f:
        yaml.dump(param, f)

param = dict(
    ## PATH
    DATASET_PATH = "/data/Datasets/VOCdevkit/VOC2012/BASIC",
    LABEL_PATH = "/data/Datasets/VOCdevkit/VOC2012/Labels/labels.txt",
    SAVE_PATH = "/data/Models/segmentation/TEST/",

    ## PARAMETERS
    BATCH_SIZE = 16,
    EPOCHS = 300,
    IMG_SIZE = 320,
    ES_PATIENT = 10,
    ONE_HOT = False,

    LR_START = 0.0001,
    LR_MAX = 0.0005,
    LR_MIN = 0.0001,
    LR_RAMPUP_EPOCHS = 4,
    LR_SUSTAIN_EPOCHS = 4,
    LR_EXP_DECAY = .8,
    
    BACKBONE_NAME = "EfficientNetB3",
    BACKBONE_TRAINABLE = True,
    FINAL_ACTIVATION = None,
    CKPT = None,

    COLORMAP = [
        [0, 0, 0], # background
        [128, 0, 0], # aeroplane
        [0, 128, 0], # bicycle
        [128, 128, 0], # bird
        [0, 0, 128], # boat
        [128, 0, 128], # bottle
        [0, 128, 128], # bus
        [128, 128, 128], # car
        [64, 0, 0], # cat
        [192, 0, 0], # chair
        [64, 128, 0], # cow
        [192, 128, 0], # diningtable
        [64, 0, 128], # dog
        [192, 0, 128], # horse
        [64, 128, 128], # motorbike
        [192, 128, 128], # person
        [0, 64, 0], # potted plant
        [128, 64, 0], # sheep
        [0, 192, 0], # sofa
        [128, 192, 0], # train
        [0, 64, 128] # tv/monitor
    ]
)