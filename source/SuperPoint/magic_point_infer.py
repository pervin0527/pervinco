import os
import cv2
import yaml
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from data import synthetic_data
from magic_point_model import MagicPoint
from synthetic_shapes import parse_primitives
from data.data_utils import photometric_augmentation, homographic_augmentation, add_keypoint_map, box_nms, downsample

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


def generate_shapes():
    drawing_primitives = [
            'draw_lines',
            'draw_polygon',
            'draw_multiple_polygons',
            'draw_ellipses',
            'draw_star',
            'draw_checkerboard',
            'draw_stripes',
            'draw_cube',
            'gaussian_noise'
    ]
    primitives = parse_primitives(config["data"]['primitives'], drawing_primitives)
    while True:
        primitive = np.random.choice(primitives)
        image = synthetic_data.generate_background(config["data"]['generation']['image_size'], **config["data"]['generation']['params']['generate_background'])
        points = np.array(getattr(synthetic_data, primitive)(image, **config["data"]['generation']['params'].get(primitive, {})))
        yield (np.expand_dims(image, axis=-1).astype(np.float32), np.flip(points.astype(np.float32), 1))

    

def build_test_dataset(path):
    dataset = tf.data.Dataset.from_generator(generate_shapes, (tf.float32, tf.float32),
                                                (tf.TensorShape(config["data"]["generation"]["image_size"] + [1]), tf.TensorShape([None, 2])))
    dataset = dataset.map(lambda i, c : downsample(i, c, **config["data"]["preprocessing"]))
    dataset = dataset.map(lambda image, keypoints : {"image" : image, "keypoints" : keypoints})

    if config["data"]["augmentation"]["photometric"]["enable"]:
        dataset = dataset.map(lambda x : photometric_augmentation(x, config))
    if config["data"]["augmentation"]["homographic"]["enable"]:
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

    testset = build_test_dataset(config["path"]["data_path"])
    test_iterator = iter(testset)

    save_path = '/'.join(config["path"]["ckpt_path"].split('/')[:-1]) + '/inference'
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    
    pbar = tqdm(total=config["model"]["test_iter"])
    for idx, data in enumerate(testset.take(config["model"]["test_iter"])):
        pred_logits, pred_probs = model(data["image"])
        nms_prob = tf.map_fn(lambda p : box_nms(p, config["model"]["nms_size"], threshold=config["model"]["threshold"], keep_top_k=0), pred_probs)

        image = (data["image"][0].numpy() * 255).astype(np.int32)
        result_image = draw_keypoints(image, np.where(nms_prob[0] > config["model"]["threshold"]), (0, 255, 0))
        
        # cv2.imshow("result", result_image)
        # cv2.waitKey(0)
        cv2.imwrite(f"{save_path}/{idx:>04}.png", result_image)
        pbar.update(1)
    
    print("DONE")