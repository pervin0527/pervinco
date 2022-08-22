import os
import cv2
import yaml
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from glob import glob
from datetime import datetime
from IPython.display import clear_output
from magic_point_model import MagicPoint
from model.angular_grad import AngularGrad
from data.data_utils import add_dummy_valid_mask, photometric_augmentation, homographic_augmentation, add_keypoint_map, box_nms

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


def build_tf_dataset(path, target="training"):
    images = sorted(glob(f"{path}/{target}/images/*.png"))
    points = sorted(glob(f"{path}/{target}/points/*.npy"))
    print(len(images), len(points))

    dataset = tf.data.Dataset.from_tensor_slices((images, points))
    dataset = dataset.map(lambda image, points : (read_image(image), tf.numpy_function(read_points, [points], tf.float32)))
    dataset = dataset.map(lambda image, points : (image, tf.reshape(points, [-1, 2])))
    dataset = dataset.shuffle(config["model"]["train_iter"])

    if target == "training":
        dataset = dataset.take(config["model"]["train_iter"])

    dataset = dataset.map(lambda image, keypoints : {"image" : image, "keypoints" : keypoints})
    dataset = dataset.map(add_dummy_valid_mask)

    if target == "training":
        dataset = dataset.map(lambda x : photometric_augmentation(x, config))
        dataset = dataset.map(lambda x : homographic_augmentation(x, config))
        
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

        nms_prob = tf.map_fn(lambda p : box_nms(p, config["model"]["nms_size"], threshold=config["model"]["threshold"], keep_top_k=0), pred_probs)
        result_img = draw_keypoints(image, np.where(nms_prob[0] > config["model"]["threshold"]), (0, 255, 0))
        # result_img = draw_keypoints(image, np.where(pred_probs[0] > config["model"]["threshold"]), (0, 255, 0))
        cv2.imwrite(f"./on_epoch_end/nms_result_{index}.png", result_img)


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
    config_path = "./configs/magic-point_shapes.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    data_path = config["path"]["dataset"] + "/synthetic_shapes_" + config["data"]["suffix"]
    print("DATASET PATH : ", data_path)
    train_dataset = build_tf_dataset(data_path, "training")
    valid_dataset = build_tf_dataset(data_path, "validation")
    test_dataset = build_tf_dataset(data_path, "test")

    show_sample(train_dataset, 10, "train")
    show_sample(valid_dataset, 10, "valid")
    show_sample(test_dataset, 5, "test")

    if config["model"]["optimizer"] == "adam":
        optimizer = tf.keras.optimizers.Adam()
    elif config["model"]["optimizer"] == "angular":
        optimizer = AngularGrad(method_angle="cos")

    clr = tfa.optimizers.CyclicalLearningRate(initial_learning_rate=config["model"]["init_lr"],
                                              maximal_learning_rate=config["model"]["max_lr"],
                                              scale_fn=lambda x : 1.0,
                                              step_size=config["model"]["epochs"] / 2)

    save_folder = datetime.now().strftime("%Y.%m.%d_%H:%M")
    save_path = config["path"]["save_path"] + f"/{save_folder}"
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    with open(f"{save_path}/train_magic-point.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    callbacks = [
        DisplayCallback(),
        tf.keras.callbacks.LearningRateScheduler(clr),
        tf.keras.callbacks.ModelCheckpoint(f"{save_path}/weights.h5", monitor="val_loss", verbose=1, save_best_only=True, save_weights_only=True)
    ]

    with strategy.scope():
        model = MagicPoint(config["model"]["backbone"], config["model"]["input_shape"], config["model"]["nms_size"], config["model"]["threshold"], config["model"]["focal_loss"], config["model"]["summary"])
        for index in range(len(model.layers)):
            model.layers[index].trainable = True
        model.compile(optimizer=optimizer)

    model.fit(train_dataset,
              validation_data=valid_dataset,
              epochs=config["model"]["epochs"],
              callbacks=callbacks)