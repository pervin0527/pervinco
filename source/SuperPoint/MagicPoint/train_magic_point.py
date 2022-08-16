import os
import sys
import cv2
import yaml
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


def homographic_augmentation(data):
    params = config["data"]["augmentation"]["homographic"]["params"]
    valid_border_margin = config["data"]["augmentation"]["homographic"]["valid_border_margin"]

    image_shape = tf.shape(data["image"])[:2]
    homography = sample_homography(image_shape, params)[0]
    warped_image = tfa.image.transform(data['image'], homography, interpolation='BILINEAR')
    valid_mask = compute_valid_mask(image_shape, homography, valid_border_margin)

    warped_points = warp_points(data['keypoints'], homography)
    warped_points = filter_points(warped_points, image_shape)

    ret = {**data, 'image': warped_image, 'keypoints': warped_points, 'valid_mask': valid_mask}
    return ret


def build_tf_dataset(path, target="training"):
    images = sorted(glob(f"{path}/{target}/images/*.png"))
    points = sorted(glob(f"{path}/{target}/points/*.npy"))
    print(len(images), len(points))

    dataset = tf.data.Dataset.from_tensor_slices((images, points))
    dataset = dataset.map(lambda image, points : (read_image(image), tf.numpy_function(read_points, [points], tf.float32)))
    dataset = dataset.map(lambda image, points : (image, tf.reshape(points, [-1, 2])))
    dataset = dataset.shuffle(train_iter)

    if target == "training":
        dataset = dataset.take(train_iter)

    dataset = dataset.map(lambda image, keypoints : {"image" : image, "keypoints" : keypoints})
    dataset = dataset.map(add_dummy_valid_mask)

    if target == "training":
        dataset = dataset.map(lambda x : photometric_augmentation(x))
        dataset = dataset.map(lambda x : homographic_augmentation(x))
        
    dataset = dataset.map(lambda x : add_keypoint_map(x))
    dataset = dataset.map(lambda d : {**d, "image" : tf.cast(d["image"], tf.float32) / 255.})
    dataset = dataset.batch(batch_size=1)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def draw_keypoints(img, corners, color):
    keypoints = [cv2.KeyPoint(int(c[1]), int(c[0]), 1) for c in np.stack(corners).T]
    return cv2.drawKeypoints(img.astype(np.uint8), keypoints, None, color=color)


def plot_predictions(model):
    if not os.path.isdir("./on_epoch_end"):
        os.makedirs("./on_epoch_end")

    for index, data in enumerate(test_dataset.take(5)):
        pred_logits, pred_probs = model.predict(data["image"])
        image = (data["image"][0].numpy() * 255).astype(np.int32)

        pred_probs = pred_probs[0]
        pred_probs[pred_probs > threshold] = 255
        cv2.imwrite(f"./on_epoch_end/keypoints_{index}.png", pred_probs)
        
        results = np.greater_equal(pred_probs, threshold).astype(np.int32)
        result_img = draw_keypoints(image, np.where(results), (0, 255, 0))
        cv2.imwrite(f"./on_epoch_end/result_{index}.png", result_img)


class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        plot_predictions(model=model)


def show_sample(dataset, n_samples, name):
    if not os.path.isdir(f"./samples/{name}"):
        os.makedirs(f"./samples/{name}")

        for index, data in enumerate(dataset.take(n_samples)):
            image = data["image"][0].numpy() ### shape : 120, 160, 1
            keypoints = data["keypoints"][0].numpy() # shape : (120, 160) values : 0 or 1
            valid_mask = data["valid_mask"][0].numpy() ## shape : (120, 160) values : 0 or 1
            keypoint_map = data["keypoint_map"][0].numpy()

            sample = draw_keypoints(image[..., 0] * 255, np.where(keypoint_map), (0, 255, 0))
            cv2.imwrite(f"./samples/{name}/{index}.png", sample)
    else:
        pass


if __name__ == "__main__":
    data_path = "/home/ubuntu/Datasets/synthetic_shapes/synthetic_shapes_v0"
    config_path = "./magic-point_shapes.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    epochs = 10000
    threshold = 0.2
    learning_rate = 0.001
    input_shape = (120, 160, 1)
    train_iter = 10000

    train_dataset = build_tf_dataset(data_path, "training")
    valid_dataset = build_tf_dataset(data_path, "validation")
    test_dataset = build_tf_dataset(data_path, "test")

    show_sample(train_dataset, 10, "train")
    show_sample(valid_dataset, 10, "valid")
    show_sample(test_dataset, 5, "test")

    optimizer = tf.keras.optimizers.Adam(learning_rate)
    cdr = tf.keras.optimizers.schedules.CosineDecayRestarts(initial_learning_rate=learning_rate,
                                                            first_decay_steps=300,
                                                            t_mul=2.0,
                                                            m_mul=0.9,
                                                            alpha=0.001)

    clr = tfa.optimizers.CyclicalLearningRate(initial_learning_rate=0.0001,
                                              maximal_learning_rate=0.009,
                                              scale_fn=lambda x : 1.0,
                                              step_size=epochs/2)

    callbacks = [DisplayCallback(),
                 tf.keras.callbacks.LearningRateScheduler(clr),
                 tf.keras.callbacks.ModelCheckpoint("/home/ubuntu/Models/MagicPoint/ckpt.h5", monitor="val_loss", verbose=1, save_best_only=True, save_weights_only=True)]

    model = MagicPoint(input_shape, summary=True)
    model.compile(optimizer=optimizer)
    for index in range(len(model.layers)):
        model.layers[index].trainable = True

    model.fit(train_dataset,
              validation_data=valid_dataset,
              epochs=epochs,
              callbacks=callbacks)