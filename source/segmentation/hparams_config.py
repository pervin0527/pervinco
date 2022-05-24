import os
import yaml
import numpy as np
import pandas as pd

data_dir = "/data/Datasets/VOCdevkit/VOC2012"
save_dir = "/data/Models/segmentation"
folder = "BASIC"
backbone_name = "ResNet101"
# checkpoint_dir = f"{save_dir}/VOC2012-BASIC-ResNet50/best.ckpt"
checkpoint_dir = None

batch_size = 16
epochs = 100
image_size = 320
early_stopping_patient = 10

original_output = True
backbone_trainable = True
one_hot_encoding = True
final_activation = "softmax"

include_class_weight = False
learning_rate = 0.0001

colormap = [
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

param = dict(
    BATCH_SIZE = batch_size,
    EPOCHS = epochs,
    IMG_SIZE = image_size,
    ES_PATIENT = early_stopping_patient,
    ONE_HOT = one_hot_encoding,
    FINAL_ACTIVATION = final_activation,
    BACKBONE_NAME = backbone_name,
    BACKBONE_TRAINABLE = backbone_trainable,
    ORIGINAL_OUTPUT = original_output,

    DATASET_PATH = f"{data_dir}/{folder}",
    LABEL_PATH = f"{data_dir}/Labels/labels.txt",
    SAVE_PATH = f"{save_dir}/{data_dir.split('/')[-1]}-{folder}-{backbone_name}",
    CKPT = checkpoint_dir,

    LR_START = learning_rate,
    LR_MAX = learning_rate * 5,
    LR_MIN = learning_rate,
    LR_RAMPUP_EPOCHS = 4,
    LR_SUSTAIN_EPOCHS = 4,
    LR_EXP_DECAY = .8,
    
    INCLUDE_CLASS_WEIGHT = include_class_weight,
    COLORMAP = colormap
)

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

    with open(param["SAVE_PATH"] + "/config.yaml", "w") as f:
        yaml.dump(param, f)