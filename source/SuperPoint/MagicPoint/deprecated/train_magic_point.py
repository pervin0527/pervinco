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


def build_tf_dataset(path, is_train):
    images = sorted(glob(f"{path}/images/*.png"))
    points = sorted(glob(f"{path}/points/*.npy"))

    dataset = tf.data.Dataset.from_tensor_slices((images, points))
    dataset = dataset.map(lambda image, points : (read_image(image), tf.numpy_function(read_points, [points], tf.float32)))
    dataset = dataset.map(lambda image, points : (image, tf.reshape(points, [-1, 2])))
    dataset = dataset.map(lambda image, keypoints : {"image" : image, "keypoints" : keypoints})
    dataset = dataset.map(add_dummy_valid_mask)

    if is_train:
        dataset = dataset.map(lambda x : photometric_augmentation(x))
        dataset = dataset.map(lambda x : homographic_augmentation(x))
        
    dataset = dataset.map(lambda x : add_keypoint_map(x))
    dataset = dataset.map(lambda d : {**d, "image" : tf.cast(d["image"], tf.float32) / 255.})
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def draw_keypoints(img, corners, color):
    keypoints = [cv2.KeyPoint(int(c[1]), int(c[0]), 1) for c in np.stack(corners).T]
    return cv2.drawKeypoints(img.astype(np.uint8), keypoints, None, color=color)


def plot_predictions(model):
    test_images = glob(f"{test_path}/images/*.png")

    idx = np.random.randint(0, len(test_images)-1)
    test_img = read_image(test_images[idx])
    # test_tensor = test_img / 255.
    test_tensor = tf.expand_dims(test_img, axis=0)
    pred_logits, pred_probs = model.predict(test_tensor)

    pred_probs = pred_probs[0]
    pred_probs = tf.cast(tf.greater_equal(pred_probs, 0.9), tf.int32)
    
    print(pred_probs)
    pred_result = draw_keypoints(test_img.numpy(), pred_probs.numpy(), color=(0, 255, 0))
    cv2.imwrite("./result.png", pred_result)


class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        plot_predictions(model=model)


if __name__ == "__main__":
    epochs = 20000
    batch_size = 1
    learning_rate = 0.001
    input_shape = (120, 160, 1)

    train_path = "/data/Datasets/Synthetic_shapes/train"
    valid_path = "/data/Datasets/Synthetic_shapes/valid"
    test_path = "/data/Datasets/Synthetic_shapes/test"

    train_dataset = build_tf_dataset(train_path, True)
    valid_dataset = build_tf_dataset(valid_path, False)

    # for data in train_dataset.take(5):
    #     image = data["image"][0].numpy()
    #     keypoints = data["keypoints"][0].numpy()
    #     valid_mask = data["valid_mask"][0].numpy()
    #     keypoint_map = data["keypoint_map"][0].numpy()

    #     print(valid_mask.shape) ## shape : (120, 160) values : 0 or 1
    #     print(keypoint_map.shape) ## shape : (120, 160) values : 0 or 1
    #     sample = draw_keypoints(image[..., 0] * 255, np.where(keypoint_map), (0, 255, 0))            
    #     cv2.imshow("sample", sample)
    #     cv2.waitKey(0)

    optimizer = tf.keras.optimizers.Adam(learning_rate)
    callbacks = [DisplayCallback(),
                 tf.keras.callbacks.ModelCheckpoint("/data/Models/MagicPoint/ckpt.h5", monitor="val_loss", verbose=1, save_best_only=True, save_weights_only=True)]

    model = MagicPoint(input_shape)
    model.compile(optimizer=optimizer)
    model.fit(train_dataset,
              validation_data=valid_dataset,
              epochs=epochs,
              callbacks=callbacks)