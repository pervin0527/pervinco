import os
import sys
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import photometric_augmentation as photaug

from glob import glob
from IPython.display import clear_output
from magic_point_model import MagicPoint
from generate_synthetic_shapes import get_data
from homograhic_augmentation import sample_homography, compute_valid_mask, warp_points, filter_points

np.set_printoptions(threshold=sys.maxsize)
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


def generate_shape():
    while True:
        image, points = get_data(image_size, resize, blur_size)
        yield (np.expand_dims(image, axis=-1).astype(np.float32), np.flip(points.astype(np.float32), 1))


def read_image(file_name):
    image = tf.io.read_file(file_name)
    image = tf.io.decode_png(image, channels=1)

    return tf.cast(image, tf.float32)


def read_points(file_name):
    return np.load(file_name.decode("utf-8")).astype(np.float32)


def add_keypoint_map(data):
    image_shape = tf.shape(data['image'])[:2]
    kp = tf.minimum(tf.cast(tf.round(data['keypoints']), tf.int32), image_shape-1)
    kmap = tf.scatter_nd(kp, tf.ones([tf.shape(kp)[0]], dtype=tf.int32), image_shape)
    
    return {**data, 'keypoint_map': kmap}


def add_dummy_valid_mask(data):
    valid_mask = tf.ones(tf.shape(data['image'])[:2], dtype=tf.int32)
    return {**data, 'valid_mask': valid_mask}


def photometric_augmentation(data):
    primitives = ['random_brightness', 'random_contrast', 'additive_speckle_noise',
                  'additive_gaussian_noise', 'additive_shade', 'motion_blur']

    params = {"random_brightness": {"max_abs_change": 75},
              "random_contrast" : {"strength_range": [0.3, 1.8]},
              "additive_gaussian_noise" : {"stddev_range": [0, 15]},
              "additive_speckle_noise" : {"prob_range" : [0, 0.0035]},
              "additive_shade" : {"transparency_range": [-0.5, 0.8],
                                  "kernel_size_range" : [50, 100]},
              "motion_blur" : {"max_kernel_size" : 7}
    }

    prim_configs = [params.get(p, {}) for p in primitives]

    indices = tf.range(len(primitives))
    def step(i, image):
        fn_pairs = [(tf.equal(indices[i], j), lambda p=p, c=c: getattr(photaug, p)(image, **c)) for j, (p, c) in enumerate(zip(primitives, prim_configs))]
        image = tf.case(fn_pairs)
        return i + 1, image

    _, image = tf.while_loop(lambda i, image: tf.less(i, len(primitives)), step, [0, data['image']], parallel_iterations=1)

    return {**data, 'image': image}


def homographic_augmentation(data):
    params = {
        "translation" : True,
        "rotation" : True,
        "scaling" : True,
        "perspective" : True,
        "scaling_amplitude" : 0.2,
        "perspective_amplitude_x" : 0.2,
        "perspective_amplitude_y" : 0.2,
        "patch_ratio" : 0.8,
        "max_angle" : 1.57,
        "allow_artifacts" : True,
        "translation_overflow" : 0.05,
    }
    valid_border_margin = 2

    image_shape = tf.shape(data["image"])[:2]
    homography = sample_homography(image_shape, params)[0]
    warped_image = tfa.image.transform(data['image'], homography, interpolation='BILINEAR')
    valid_mask = compute_valid_mask(image_shape, homography, valid_border_margin)

    warped_points = warp_points(data['keypoints'], homography)
    warped_points = filter_points(warped_points, image_shape)

    ret = {**data, 'image': warped_image, 'keypoints': warped_points, 'valid_mask': valid_mask}
    return ret


def downsample(image, coordinates):
    k_size = blur_size
    kernel = cv2.getGaussianKernel(k_size, 0)[:, 0]
    kernel = np.outer(kernel, kernel).astype(np.float32)
    kernel = tf.reshape(tf.convert_to_tensor(kernel), [k_size]*2+[1, 1])
    pad_size = int(k_size/2)
    image = tf.pad(image, [[pad_size]*2, [pad_size]*2, [0, 0]], 'REFLECT')
    image = tf.expand_dims(image, axis=0)  # add batch dim
    image = tf.nn.depthwise_conv2d(image, kernel, [1, 1, 1, 1], 'VALID')[0]

    ratio = tf.divide(tf.convert_to_tensor(resize), tf.shape(image)[0:2])
    coordinates = coordinates * tf.cast(ratio, tf.float32)
    image = tf.image.resize(image, resize, method=tf.image.ResizeMethod.BILINEAR)

    return image, coordinates


def build_tf_dataset(num_examples, is_train):
    dataset = tf.data.Dataset.from_generator(generate_shape, (tf.float32, tf.float32), (tf.TensorShape(image_size + [1]), tf.TensorShape([None, 2])))
    dataset = dataset.map(lambda img, coord : downsample(img, coord))
    dataset = dataset.map(lambda image, points : (image, tf.reshape(points, [-1, 2])))
    dataset = dataset.take(num_examples)
    dataset = dataset.map(lambda image, keypoints : {"image" : image, "keypoints" : keypoints})
    dataset = dataset.map(add_dummy_valid_mask)

    if is_train:
        dataset = dataset.map(lambda x : photometric_augmentation(x))
        dataset = dataset.map(lambda x : homographic_augmentation(x))
        
    dataset = dataset.map(lambda x : add_keypoint_map(x))
    dataset = dataset.map(lambda d : {**d, "image" : tf.cast(d["image"], tf.float32) / 255.})
    dataset = dataset.batch(1)

    return dataset


def draw_keypoints(img, corners, color):
    keypoints = [cv2.KeyPoint(int(c[1]), int(c[0]), 1) for c in np.stack(corners).T]
    return cv2.drawKeypoints(img.astype(np.uint8), keypoints, None, color=color)


def plot_predictions(model):
    if not os.path.isdir("./on_epoch_end"):
        os.makedirs("./on_epoch_end")

    for index, data in enumerate(test_dataset):
        pred_logits, pred_probs = model.predict(data["image"])

        image = (data["image"][0].numpy() * 255).astype(np.int32)
        keypoints = np.greater_equal(pred_probs[0], 0.6).astype(np.int32)
        result = draw_keypoints(image, np.where(keypoints), (0, 255, 0))
        
        cv2.imwrite(f"./on_epoch_end/result_{index}.png", result)


class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        plot_predictions(model=model)


def show_sample(dataset, n_samples, name):
    if not os.path.isdir(f"./samples/{name}"):
        os.makedirs(f"./samples/{name}")

        print(f"{name} samples")
        for index, data in enumerate(dataset.take(n_samples)):
            image = data["image"][0].numpy() ### shape : 120, 160, 1
            keypoints = data["keypoints"][0].numpy() # shape : (120, 160) values : 0 or 1
            valid_mask = data["valid_mask"][0].numpy() ## shape : (120, 160) values : 0 or 1
            keypoint_map = data["keypoint_map"][0].numpy()

            sample = draw_keypoints(image[..., 0] * 255, np.where(keypoint_map), (0, 255, 0))
            cv2.imwrite(f"./samples/{name}/{index}.png", sample)
    else:
        pass
    print("")


if __name__ == "__main__":
    epochs = 10000
    batch_size = 1
    learning_rate = 0.001
    input_shape = (120, 160, 1)

    blur_size = 21
    image_size = [960, 1280]
    resize = [120, 160]

    train_dataset = build_tf_dataset(10000, True)
    valid_dataset = build_tf_dataset(200, False)
    test_dataset = build_tf_dataset(10, False)

    show_sample(train_dataset, 10, "train")
    show_sample(valid_dataset, 10, "valid")
    show_sample(test_dataset, 5, "test")

    optimizer = tf.keras.optimizers.Adam(learning_rate)
    callbacks = [
        DisplayCallback(),
        tf.keras.callbacks.ModelCheckpoint("/data/Models/MagicPoint/ckpt.h5", monitor="val_loss", verbose=1, save_best_only=True, save_weights_only=True)]

    model = MagicPoint(input_shape, summary=False)
    model.compile(optimizer=optimizer)
    model.fit(train_dataset,
              validation_data=valid_dataset,
              epochs=epochs,
              callbacks=callbacks)