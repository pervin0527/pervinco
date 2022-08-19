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
from homograhic_augmentation import sample_homography, compute_valid_mask, warp_points, filter_points

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


def add_dummy_valid_mask(data):
    with tf.name_scope('dummy_valid_mask'):
        valid_mask = tf.ones(tf.shape(data['image'])[:2], dtype=tf.int32)
    return {**data, 'valid_mask': valid_mask}


def add_keypoint_map(data):
    image_shape = tf.shape(data['image'])[:2]
    kp = tf.minimum(tf.cast(tf.round(data['keypoints']), tf.int32), image_shape-1)
    kmap = tf.scatter_nd(kp, tf.ones([tf.shape(kp)[0]], dtype=tf.int32), image_shape)
    
    return {**data, 'keypoint_map': kmap}


def read_points(filename):
    return np.load(filename.decode("utf-8"))["points"].astype(np.float32)


def read_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_png(image, channels=3)
    return tf.cast(image, tf.float32)


def ratio_preserving_resize(image, config):
    target_size = tf.convert_to_tensor(config['resize'])
    scales = tf.cast(tf.divide(target_size, tf.shape(image)[:2]), tf.float32)
    new_size = tf.cast(tf.shape(image)[:2], tf.float32) * tf.reduce_max(scales)
    image = tf.image.resize(image, tf.cast(new_size, tf.int32), method=tf.image.ResizeMethod.BILINEAR)
    return tf.image.resize_with_crop_or_pad(image, target_size[0], target_size[1])


def preprocess(image):
    image = tf.image.rgb_to_grayscale(image)
    if config["data"]["preprocessing"]["resize"]:
        image = ratio_preserving_resize(image, config["data"]["preprocessing"])
    return image


def photometric_augmentation(data):
    primitives = config["data"]["augmentation"]["photometric"]["primitives"]
    params = config["data"]["augmentation"]["photometric"]["params"]

    prim_configs = [params.get(p, {}) for p in primitives]
    indices = tf.range(len(primitives))
    def step(i, image):
        fn_pairs = [(tf.equal(indices[i], j), lambda p=p, c=c: getattr(photaug, p)(image, **c)) for j, (p, c) in enumerate(zip(primitives, prim_configs))]
        image = tf.case(fn_pairs)
        return i + 1, image

    _, image = tf.while_loop(lambda i, image: tf.less(i, len(primitives)), step, [0, data['image']], parallel_iterations=1)

    return {**data, 'image': image}


def homographic_augmentation(data, config, add_homography=False):
    params = config["data"]["augmentation"]["homographic"]["params"]
    valid_border_margin = config["data"]["augmentation"]["homographic"]["valid_border_margin"]

    image_shape = tf.shape(data["image"])[:2]
    homography = sample_homography(image_shape, params)[0]
    warped_image = tfa.image.transform(data['image'], homography, interpolation='BILINEAR')
    valid_mask = compute_valid_mask(image_shape, homography, valid_border_margin)

    warped_points = warp_points(data['keypoints'], homography)
    warped_points = filter_points(warped_points, image_shape)

    ret = {**data, 'image': warped_image, 'keypoints': warped_points, 'valid_mask': valid_mask}
    if add_homography:
        ret["homography"] = homography
    return ret


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


def flat2mat(H):
    return tf.reshape(tf.concat([H, tf.ones([tf.shape(H)[0], 1])], axis=1), [-1, 3, 3])


def mat2flat(H):
    H = tf.reshape(H, [-1, 9])
    return (H / H[:, 8:9])[:, :8]


def invert_homography(H):
    return mat2flat(tf.linalg.inv(flat2mat(H)))


def homography_adaptation(image, model, config):
    logits, probs = model(image)
    counts = tf.ones_like(probs)
    image = image

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
            count = tf.nn.erosion2d(count, tf.cast(tf.constant(kernel)[..., tf.newaxis], tf.float32),
                                    [1, 1, 1, 1], [1, 1, 1, 1], "SAME")[..., 0] + 1.
            mask = tf.nn.erosion2d(mask, tf.cast(tf.constant(kernel)[..., tf.newaxis], tf.float32),
                                   [1, 1, 1, 1], [1, 1, 1, 1], "SAME")[..., 0] + 1.

        logits, prob = model(warped)
        prob = prob * mask
        prob_proj = H_transform(tf.expand_dims(prob, -1), H_inv, interpolation="BILNEAR")[..., 0]
        prob_proj = prob_proj * count

        probs = tf.concat([probs, tf.expand_dims(prob_proj, -1)], axis=-1)
        counts = tf.concat([counts, tf.expand_dims(count, -1)], axis=-1)
        images = tf.concat([images, tf.expand_dims(warped, -1)], axis=-1)
        return i + 1, probs, counts, images

    _, probs, counts, images = tf.while_loop(lambda i, p, c, im : tf.less(i, config["num"] - 1), step, [0, probs, counts, images],
                                             parallel_iterations=1, back_prop=False,
                                             shape_invariants=[tf.TensorShape([]),
                                                               tf.TensorShape([None, None, None, None]),
                                                               tf.TensorShape([None, None, None, None]),
                                                               tf.TensorShape([None, None, None, 1, None])])
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
    print(files["image_paths"][:3])
    print(files["names"][:3])

    dataset = make_tf_dataset(files, False)
    for data in dataset.take(1):
        print(data)

    model = MagicPoint(config["model"]["backbone_name"], config["model"]["input_shape"], config["model"]["nms_size"], config["model"]["threshold"], False)
    model.built = True
    model.load_weights(config["model"]["ckpt_path"])
    print("model_loaded")