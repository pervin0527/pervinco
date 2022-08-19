import os
import cv2
import yaml
import numpy as np
import tensorflow as tf

from glob import glob
from magic_point_model import MagicPoint

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# np.set_printoptions(threshold=sys.maxsize)
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
    config_path = "./magic-point_coco_export.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model = MagicPoint(config["model"]["backbone_name"], config["model"]["input_shape"], config["model"]["nms_size"], config["model"]["threshold"], False)
    model.built = True
    model.load_weights(config["path"]["ckpt_path"])
    print("model_loaded")

    images = sorted(glob(config["path"]["coco_path"]) + "/*.jpg")
    print(len(images))