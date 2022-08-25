import os
import sys
import cv2
import yaml
import numpy as np
import tensorflow as tf

from magic_point_model import MagicPoint
from data.data_utils import homography_adaptation, box_nms

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
np.set_printoptions(threshold=sys.maxsize)
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


def draw_keypoints(img, corners, color):
    keypoints = [cv2.KeyPoint(int(c[1]), int(c[0]), 1) for c in np.stack(corners).T]
    return cv2.drawKeypoints(img.astype(np.uint8), keypoints, None, color=color)


if __name__ == "__main__":
    config_path = "./configs/magic-point_coco_sample.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model = MagicPoint(config["model"]["input_shape"], config["model"]["nms_size"], config["model"]["threshold"])
    model.built = True
    model.load_weights(config["path"]["ckpt_path"])

    image = tf.io.read_file(config["path"]["image_path"])
    image = tf.io.decode_jpeg(image, channels=1)
    image = tf.image.resize(image, config["data"]["preprocessing"]["resize"])
    input_image = tf.cast(image, tf.float32) / 255.

    outputs = homography_adaptation(input_image, model, config)
    outputs ={k: v[0] for k, v in outputs.items()}  # batch to single element
    outputs['prob_nms'] = box_nms(outputs['prob'], config["model"]['nms_size'], keep_top_k=config["model"]['top_k'])
    outputs['pred'] = tf.cast(tf.greater_equal(outputs['prob_nms'], config["model"]['threshold']), dtype=np.int32)

    image = image.numpy()
    pred = outputs["pred"].numpy()
    result = draw_keypoints(image, np.where(pred), (0, 255, 0))

    if not os.path.isdir("./samples/coco_export"):
        os.makedirs("./samples/coco_export")

    cv2.imwrite("./samples/coco_export/image.jpg", image)
    cv2.imwrite("./samples/coco_export/result.jpg", result)