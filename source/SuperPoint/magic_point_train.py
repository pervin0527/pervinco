import os
import cv2
import yaml
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from glob import glob
from datetime import datetime
from data import synthetic_data
from IPython.display import clear_output
from magic_point_model import MagicPoint
from model.angular_grad import AngularGrad
from synthetic_shapes import parse_primitives
from data.data_utils import add_dummy_valid_mask, photometric_augmentation, homographic_augmentation, add_keypoint_map, box_nms, downsample


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


def build_tf_dataset(path, target="training"):
    if not config["model"]["on-the-fly"]:
        images = sorted(glob(f"{path}/{target}/images/*.png"))
        points = sorted(glob(f"{path}/{target}/points/*.npy"))
        print(len(images), len(points))

        dataset = tf.data.Dataset.from_tensor_slices((images, points))
        dataset = dataset.map(lambda image, points : (read_image(image), tf.numpy_function(read_points, [points], tf.float32)), num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(lambda image, points : (image, tf.reshape(points, [-1, 2])), num_parallel_calls=tf.data.AUTOTUNE)
        
    else:
        dataset = tf.data.Dataset.from_generator(generate_shapes, (tf.float32, tf.float32), (tf.TensorShape(config["data"]["generation"]["image_size"] + [1]), tf.TensorShape([None, 2])))
        dataset = dataset.map(lambda i, c : downsample(i, c, **config["data"]["preprocessing"]), num_parallel_calls=tf.data.AUTOTUNE)

    if target == "training":
        # dataset = dataset.take(config["model"]["train_iter"])
        steps = int(tf.math.ceil(config["model"]["train_iter"] / config["model"]["batch_size"]))
        
    elif target == "validation":
        dataset = dataset.take(config["model"]["valid_iter"])
        steps = int(tf.math.ceil(config["model"]["valid_iter"] / config["model"]["batch_size"]))

    elif target == "test":
        dataset = dataset.take(config["model"]["test_iter"])
        config["model"]["batch_size"] = 1
        steps = 0

    dataset = dataset.map(lambda image, keypoints : {"image" : image, "keypoints" : keypoints}, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(add_dummy_valid_mask, num_parallel_calls=tf.data.AUTOTUNE)

    if target == "training":
        if config["data"]["augmentation"]["photometric"]["enable"]:
            dataset = dataset.map(lambda x : photometric_augmentation(x, config), num_parallel_calls=tf.data.AUTOTUNE)
        if config["data"]["augmentation"]["homographic"]["enable"]:
            dataset = dataset.map(lambda x : homographic_augmentation(x, config), num_parallel_calls=tf.data.AUTOTUNE)
        
    dataset = dataset.map(lambda x : add_keypoint_map(x), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(lambda d : {**d, "image" : tf.cast(d["image"], tf.float32) / 255., "keypoints" : tf.zeros([0], tf.float32)}, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.batch(batch_size=config["model"]["batch_size"])
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    dataset = dataset.repeat()
    
    return dataset, steps


def draw_keypoints(img, corners, color):
    keypoints = [cv2.KeyPoint(int(c[1]), int(c[0]), 1) for c in np.stack(corners).T]
    return cv2.drawKeypoints(img.astype(np.uint8), keypoints, None, color=color)


def plot_predictions(model):
    if not os.path.isdir(f"{save_path}/on_epoch_end"):
        os.makedirs(f"{save_path}/on_epoch_end/pred")
        os.makedirs(f"{save_path}/on_epoch_end/gt")

    for index, data in enumerate(test_dataset.take(config["model"]["test_iter"])):
        pred_logits, pred_probs = model.predict(data["image"])
        image = (data["image"][0].numpy() * 255).astype(np.int32)

        nms_prob = tf.map_fn(lambda p : box_nms(p, config["model"]["nms_size"], threshold=config["model"]["threshold"], keep_top_k=0), pred_probs)
        result_img = draw_keypoints(image, np.where(nms_prob[0] > config["model"]["threshold"]), (0, 255, 0))
        # result_img = draw_keypoints(image, np.where(pred_probs[0] > config["model"]["threshold"]), (0, 255, 0))
        cv2.imwrite(f"{save_path}/on_epoch_end/pred/pred-{index:>04}.png", result_img)

        gt_keypoint_map = data["keypoint_map"][0].numpy()
        gt_img = draw_keypoints(image, np.where(gt_keypoint_map), (0, 255, 0))
        cv2.imwrite(f"{save_path}/on_epoch_end/gt/gt-{index:>04}.png", gt_img)
        

class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        plot_predictions(model=model)


if __name__ == "__main__":
    config_path = "./configs/magic-point_train.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    data_path = config["path"]["dataset"] + "/synthetic_shapes_" + config["data"]["suffix"]
    print("DATASET PATH : ", data_path)
    train_dataset, train_steps = build_tf_dataset(data_path, "training")
    valid_dataset, valid_steps = build_tf_dataset(data_path, "validation")
    test_dataset, _ = build_tf_dataset(data_path, "test")

    save_folder = datetime.now().strftime("%Y_%m_%d-%H_%M")
    save_path = config["path"]["save_path"] + f"/{save_folder}"
    print("SAVE PATH : ", save_path)

    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    if config["model"]["optimizer"] == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=config["model"]["init_lr"], beta_1=0.9, beta_2=0.999)
    elif config["model"]["optimizer"] == "angular":
        optimizer = AngularGrad(method_angle="cos", learning_rate=config["model"]["init_lr"])

    clr = tfa.optimizers.CyclicalLearningRate(initial_learning_rate=config["model"]["init_lr"],
                                              maximal_learning_rate=config["model"]["max_lr"],
                                              scale_fn=lambda x : 1.0,
                                              step_size=config["model"]["epochs"] / 2)

    with open(f"{save_path}/train_magic-point.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    callbacks = [
        DisplayCallback(),
        tf.keras.callbacks.LearningRateScheduler(clr),
        tf.keras.callbacks.ModelCheckpoint(f"{save_path}/weights.h5", monitor="val_loss", verbose=1, save_best_only=True, save_weights_only=True),
        tf.keras.callbacks.TensorBoard(f"{save_path}/logs", write_graph=True, write_images=True, write_steps_per_second=True, update_freq="epoch")
    ]

    with strategy.scope():
        model = MagicPoint(config["model"]["input_shape"], config["model"]["nms_size"], config["model"]["threshold"], config["model"]["summary"])
        
        if config["model"]["ckpt_path"]:
            model.built = True
            model.load_weights(config["model"]["ckpt_path"])
            print("Weight Loaded")

        # for index in range(len(model.layers)):
        #     model.layers[index].trainable = True

        model.compile(optimizer=optimizer)

    model.fit(train_dataset,
              steps_per_epoch=train_steps,
              validation_data=valid_dataset,
              validation_steps=valid_steps,
              epochs=config["model"]["epochs"],
              callbacks=callbacks,
              verbose=1)