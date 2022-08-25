import os
import cv2
import yaml
import numpy as np
import tensorflow as tf

from glob import glob
from magic_point_model import MagicPoint
from data.data_utils import photometric_augmentation, homographic_augmentation, add_keypoint_map, box_nms

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


def read_img_file(file):
    file = tf.io.read_file(file)
    image = tf.io.decode_png(file, channels=1)

    return tf.cast(image, tf.float32)


def read_pnt_file(file):
    return np.load(file.decode("utf-8")).astype(np.float32)
    

def build_test_dataset(path, augmentation):
    images = sorted(glob(f"{path}/images/*.png"))
    points = sorted(glob(f"{path}/points/*.npy"))
    print(len(images), len(points))

    dataset = tf.data.Dataset.from_tensor_slices((images, points))
    dataset = dataset.map(lambda image, points : (read_img_file(image), tf.numpy_function(read_pnt_file, [points], tf.float32)))
    dataset = dataset.map(lambda image, points : (image, tf.reshape(points, [-1, 2])))
    dataset = dataset.map(lambda image, keypoints : {"image" : image, "keypoints" : keypoints})

    if augmentation:
        dataset = dataset.map(lambda x : photometric_augmentation(x, config))
        dataset = dataset.map(lambda x : homographic_augmentation(x, config))
    
    dataset = dataset.map(lambda x : add_keypoint_map(x))
    dataset = dataset.map(lambda d : {**d, "image" : tf.cast(d["image"], tf.float32) / 255.})
    dataset = dataset.batch(1)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


def draw_keypoints(img, corners, color):
    keypoints = [cv2.KeyPoint(int(c[1]), int(c[0]), 1) for c in np.stack(corners).T]
    return cv2.drawKeypoints(img.astype(np.uint8), keypoints, None, color=color)


if __name__ == "__main__":
    config_path = "./configs/magic_point_infer.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model = MagicPoint(config["model"]["input_shape"], config["model"]["nms_size"], config["model"]["threshold"])
    model.built = True
    model.load_weights(config["path"]["ckpt_path"])
    print("Model Loaded")

    testset = build_test_dataset(config["path"]["data_path"], config["data"]["augmentation"])
    test_iterator = iter(testset)

    while True:
        data = test_iterator.get_next()
        pred_logits, pred_probs = model(data["image"])
        nms_prob = tf.map_fn(lambda p : box_nms(p, config["model"]["nms_size"], threshold=config["model"]["threshold"], keep_top_k=0), pred_probs)

        image = (data["image"][0].numpy() * 255).astype(np.int32)
        result_image = draw_keypoints(image, np.where(nms_prob[0] > config["model"]["threshold"]), (0, 255, 0))
        cv2.imshow("result", result_image)
        cv2.waitKey(0)