import os
import cv2
import math
import numpy as np
import pandas as pd
import tensorflow as tf

from glob import glob
from IPython.display import clear_output
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


def get_files(dir):
    image_files = sorted(glob(f"{dir}/images/*.jpg"))
    keypoint_files = sorted(glob(f"{dir}/keypoints/*.txt"))

    print(len(image_files), len(keypoint_files))
    return image_files, keypoint_files


def data_process(images, keypoints):
    images = tf.io.read_file(images)
    images = tf.image.decode_jpeg(images, channels=3)
    images.set_shape([IMG_SIZE, IMG_SIZE, 3])

    keypoints = tf.reshape(keypoints, (1, 1, 98 * 2))
    keypoints = keypoints / IMG_SIZE
    keypoints.set_shape([1, 1, KEYPOINTS])
  
    return images, keypoints


def load_keypoints(keypoint_files):
    total_keypoints = []

    for file in keypoint_files:
        f = open(file, "r")
        keypoints = f.readline()
        keypoints = keypoints.split(',')
        keypoints = list(map(float, keypoints))

        total_keypoints.append(keypoints)

    return np.array(total_keypoints)


def get_tf_data(images, keypoints, is_train):
    dataset = tf.data.Dataset.from_tensor_slices((images, keypoints))
    dataset = dataset.map(data_process, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.repeat()
    if is_train:
        dataset = dataset.shuffle(buffer_size=len(train_image_files))
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

def custom_wing_loss(w=10.0, epsilon=2.0):
    def wing_loss(y_true, y_pred):
        x = y_true - y_pred
        c = w * (1.0 - tf.math.log(1.0 + w / epsilon))
        absolute_x = tf.abs(x)
        losses = tf.where(tf.greater(w, absolute_x), w * tf.math.log(1.0 + absolute_x / epsilon), absolute_x - c)
        loss = tf.reduce_mean(tf.reduce_sum(losses, axis=[1, 2]), axis=0)

        return loss
    return wing_loss


def build_model():
    with strategy.scope():
        base_model = tf.keras.applications.ResNet50(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet")    
        base_model.trainable = True

        input_layer = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
        x = tf.keras.applications.resnet50.preprocess_input(input_layer)
        x = base_model(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.SeparableConv2D(KEYPOINTS, kernel_size=5, strides=1, activation="relu")(x)
        output_layer = tf.keras.layers.SeparableConv2D(KEYPOINTS, kernel_size=3, strides=1, activation="sigmoid")(x)

        model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
        model.summary()

        return model


def read_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, size=[IMG_SIZE, IMG_SIZE])

    return image


def show_predictions(image_list, model):
    if not os.path.isdir("./Display_callback"):
        os.makedirs("./Display_callback")
        
    for idx, image_file in enumerate(image_list):
        result_image = cv2.imread(image_file)
        image_tensor = read_image(image_file)
        predictions = model.predict(np.expand_dims(image_tensor, axis=0), verbose=0)
        predictions = predictions.reshape(-1, 98, 2) * IMG_SIZE

        for (x, y) in predictions[0]:
            cv2.circle(result_image, (int(x), int(y)), radius=1, color=(0, 0, 255), thickness=3)

        cv2.imwrite(f"./Display_callback/{idx}.jpg", result_image)


class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        show_predictions(test_image_files[:5], model=model)


if __name__ == "__main__":
    train_dir = "/home/ubuntu/Datasets/WFLW/train"
    test_dir = "/home/ubuntu/Datasets/WFLW/test"

    EPOCHS = 300
    BATCH_SIZE = 256
    IMG_SIZE = 256
    KEYPOINTS = 98 * 2
    SAVE_DIR = "/home/ubuntu/Models/facial_landmark"

    train_image_files, train_keypoint_files = get_files(train_dir)
    test_image_files, test_keypoint_files = get_files(test_dir)

    train_keypoints = load_keypoints(train_keypoint_files)
    test_keypoints = load_keypoints(test_keypoint_files)
    print(train_keypoints.shape, test_keypoints.shape)

    train_dataset = get_tf_data(train_image_files, train_keypoints, is_train=True)
    test_dataset = get_tf_data(test_image_files, test_keypoints, is_train=False)
    print(train_dataset)
    print(test_dataset)

    train_steps_per_epoch = int(tf.math.ceil(len(train_image_files) / BATCH_SIZE).numpy())
    test_steps_per_epoch = int(tf.math.ceil(len(test_image_files) / BATCH_SIZE).numpy())

    callbacks = [DisplayCallback(),
                 tf.keras.callbacks.ModelCheckpoint(f"{SAVE_DIR}/best.ckpt", monitor="val_loss", verbose=1, mode="min", save_weights_only=True)]

    model = build_model()
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.001), loss = custom_wing_loss()) # tf.keras.losses.MeanSquaredError()

    history = model.fit(
        train_dataset,
        steps_per_epoch=train_steps_per_epoch,
        validation_data=test_dataset,
        validation_steps=test_steps_per_epoch,
        verbose=1,
        epochs=EPOCHS,
        callbacks=callbacks
    )