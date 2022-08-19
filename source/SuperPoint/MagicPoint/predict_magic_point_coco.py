import os
import sys
import cv2
import yaml
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import photometric_augmentation as photaug

from pathlib import Path
from magic_point_model import MagicPoint
from tensorflow_addons.image import transform as H_transform
from homograhic_augmentation import sample_homography
from data_utils import add_dummy_valid_mask, add_keypoint_map, ratio_preserving_resize, photometric_augmentation, homographic_augmentation, invert_homography

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


def build_dataset(data_dir, config):
    base_path = Path(data_dir, "train2014/")
    image_paths = list(base_path.iterdir())
    if config["data"]["truncate"]:
        image_paths = image_paths[:config["data"]["truncate"]]
    names = [p.stem for p in image_paths]
    image_paths = [str(p) for p in image_paths]
    files = {"image_paths" : image_paths, "names" : names}

    return files


def read_points(filename):
    return np.load(filename.decode("utf-8"))["points"].astype(np.float32)


def read_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_png(image, channels=3)
    return tf.cast(image, tf.float32)


def preprocess(image):
    image = tf.image.rgb_to_grayscale(image)
    if config["data"]["preprocessing"]["resize"]:
        image = ratio_preserving_resize(image, config["data"]["preprocessing"])
    return image


def make_tf_dataset(files, is_train):
    has_keypoints = "label_paths" in files

    names = tf.data.Dataset.from_tensor_slices(files["names"])
    images = tf.data.Dataset.from_tensor_slices(files["image_paths"])
    images = images.map(read_image)
    images = images.map(preprocess)
    data = tf.data.Dataset.zip({"image" : images, "name" : names})

    if has_keypoints:
        keypoint = tf.data.Dataset.from_tensor_slices(files["label_paths"])
        keypoint = keypoint.map(lambda path : tf.numpy_function(read_points, [path], tf.float32))
        kepoint = keypoint.map(lambda points : tf.reshape(points, [-1, 2]))
        data = tf.data.Dataset.zip((data, kepoint).map(lambda d, k : {**d, "keypoints" : k}))
        data = data.map(add_dummy_valid_mask)

    if not is_train:
        data = data.take(config["data"]["validation_size"])

    if config["data"]["warped_pair"]["enable"]:
        assert has_keypoints
        warped = data.map(lambda d : homographic_augmentation(d, config["data"]["warped_pair"], add_homography=True))

        if is_train and config["data"]["augmentatioin"]["photometric"]["enable"]:
            warped = warped.map(lambda d : d, config["data"]["augmentation"]["photometric"])

        warped = warped.map(add_keypoint_map)
        data = tf.data.Dataset.zip((data, warped))
        data = data.map(lambda d, w : {**d, "warped" : w})

    if has_keypoints and is_train:
        if config["data"]["augmentation"]["photometric"]["enable"]:
            data = data.map(lambda d : photometric_augmentation(d, config["data"]["augmentation"]["photometric"]))
        if config["data"]["augmentation"]["homographic"]["enable"]:
            assert not config["data"]["warped_pair"]["enable"]
            data = data.map(lambda d : homographic_augmentation(d, config["data"]["augmentation"]["homographic"]))

    if has_keypoints:
        data = data.map(add_dummy_valid_mask)
    
    data = data.map(lambda d : {**d, "image" : tf.cast(d["image"], tf.float32) / 255.})

    if config["data"]["warped_pair"]["enable"]:
        data = data.map(lambda d : {**d, "warped" : {**d["warped"], "image" : tf.cast(d["warped"]["image"], tf.float32) / 255.}})

    return data


def homography_adaptation(data, model, config):
    image = data["image"][0].numpy()
    image = tf.expand_dims(image, axis=0)

    logits, probs = model(image)
    counts = tf.ones_like(probs)
    images = image

    probs = tf.expand_dims(probs, axis=-1)
    counts = tf.expand_dims(counts, axis=-1)
    images = tf.expand_dims(images, axis=-1)

    shape = tf.shape(image)[1:3]
    
    def step(i, probs, counts, images):
        H = sample_homography(shape, config["model"]["homography_adaptation"]["homographies"])
        H_inv = invert_homography(H)
        warped = H_transform(image, H, interpolation='BILINEAR')
        count = H_transform(tf.expand_dims(tf.ones(tf.shape(image)[:3]), -1), H_inv, interpolation="nearest")
        mask = H_transform(tf.expand_dims(tf.ones(tf.shape(image)[:3]), -1), H, interpolation="nearest")

        if config["model"]["homography_adaptation"]["valid_border_margin"]:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (config["model"]["homography_adaptation"]["valid_border_margin"] *2,) * 2)
            count = tf.nn.erosion2d(value=count, 
                                    filters=tf.cast(tf.constant(kernel)[..., tf.newaxis], tf.float32), 
                                    strides=[1, 1, 1, 1], 
                                    dilations=[1, 1, 1, 1], 
                                    data_format="NHWC",
                                    padding="SAME")[..., 0] + 1.
            mask = tf.nn.erosion2d(value=mask,
                                   filters=tf.cast(tf.constant(kernel)[..., tf.newaxis], tf.float32),
                                   strides=[1, 1, 1, 1],
                                   dilations=[1, 1, 1, 1],
                                   data_format="NHWC",
                                   padding="SAME")[..., 0] + 1.

        logits, prob = model(warped)
        prob = prob * mask
        prob_proj = H_transform(tf.expand_dims(prob, -1), H_inv, interpolation="BILNEAR")[..., 0]
        prob_proj = prob_proj * count

        probs = tf.concat([probs, tf.expand_dims(prob_proj, -1)], axis=-1)
        counts = tf.concat([counts, tf.expand_dims(count, -1)], axis=-1)
        images = tf.concat([images, tf.expand_dims(warped, -1)], axis=-1)
        return i + 1, probs, counts, images

    _, probs, counts, images = tf.nest.map_structure(tf.stop_gradient, 
                                                     tf.while_loop(lambda i, p, c, im : tf.less(i, config["model"]["homography_adaptation"]["num"] - 1), 
                                                                   step,
                                                                   [0, probs, counts, images],
                                                                   parallel_iterations=1,
                                                                   shape_invariants=[tf.TensorShape([]),
                                                                                   tf.TensorShape([None, None, None, None]),
                                                                                   tf.TensorShape([None, None, None, None]),
                                                                                   tf.TensorShape([None, None, None, 1, None])]))
    counts = tf.reduce_sum(counts, axis=-1)
    max_prob = tf.reduce_sum(probs, axis=-1)
    mean_prob = tf.reduce_sum(probs, axis=-1) / counts

    if config["model"]["homography_adaptation"]["aggregation"] == "max":
        prob = max_prob
    elif config["model"]["homography_adaptation"]["aggregation"] == "sum":
        prob = mean_prob
    else:
        raise ValueError('Unkown aggregation method: {}'.format(config['aggregation']))

    if config['model']['homography_adaptation']['filter_counts']:
        prob = tf.where(tf.greater_equal(counts, config['filter_counts']), prob, tf.zeros_like(prob))

    return {'prob': prob, 'counts': counts, 'mean_prob': mean_prob, 'input_images': images, 'H_probs': probs}  # debug

        
if __name__ == "__main__":
    coco_path = "/home/ubuntu/Datasets/COCO2014"
    config_path = "./magic-point_coco_export.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    files = build_dataset(coco_path, config) ## keys : "image_paths", "names"
    print(len(files["image_paths"]), len(files["names"]))

    dataset = make_tf_dataset(files, False)
    iterator = iter(dataset)
    # for data in dataset.take(1):
    #     image, name = data["image"].numpy(), data["name"].numpy()
    #     print(image.shape, name)

    model = MagicPoint(config["model"]["backbone_name"], config["model"]["input_shape"], config["model"]["nms_size"], config["model"]["threshold"], False)
    model.built = True
    model.load_weights(config["model"]["ckpt_path"])
    print("model_loaded")

    while True:
        data = []
        try:
            for _ in range(config["model"]["batch_size"]):
                data.append(iterator.get_next())
        except (StopIteration, tf.errors.OutOfRangeError):
            if not data:
                break
            data += [data[-1] for _ in range(config["model"]["batch_size"] - len(data))]
        data = dict(zip(data[0], zip(*[d.values() for d in data])))

        prediction = homography_adaptation(data, model, config)
        print(prediction)
        print("???")