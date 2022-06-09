import os
import numpy as np
import pandas as pd
import tensorflow as tf

from glob import glob

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

    keypoints.set_shape([KEYPOINTS])
  
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


def get_tf_data(images, keypoints):
    dataset = tf.data.Dataset.from_tensor_slices((images, keypoints))
    dataset = dataset.map(data_process, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.repeat()
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def build_model():
    base_model = tf.keras.applications.ResNet50(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet")    
    base_model.trainable = True

    input_layer = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model(input_layer)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    output_layer = tf.keras.layers.Dense(KEYPOINTS)(x)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    model.summary()

    return model


if __name__ == "__main__":
    train_dir = "/data/Datasets/WFLW/train"
    test_dir = "/data/Datasets/WFLW/test"

    EPOCHS = 100
    BATCH_SIZE = 64
    IMG_SIZE = 112
    KEYPOINTS = 98 * 2

    train_image_files, train_keypoint_files = get_files(train_dir)
    test_image_files, test_keypoint_files = get_files(test_dir)

    train_keypoints = load_keypoints(train_keypoint_files)
    test_keypoints = load_keypoints(test_keypoint_files)
    print(train_keypoints.shape, test_keypoints.shape)

    train_dataset = get_tf_data(train_image_files, train_keypoints)
    test_dataset = get_tf_data(test_image_files, test_keypoints)
    print(train_dataset)
    print(test_dataset)

    train_steps_per_epoch = int(tf.math.ceil(len(train_image_files) / BATCH_SIZE).numpy())
    test_steps_per_epoch = int(tf.math.ceil(len(test_image_files) / BATCH_SIZE).numpy())

    model = build_model()
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    history = model.fit(
        train_dataset,
        steps_per_epoch=train_steps_per_epoch,
        validation_data=test_dataset,
        validation_steps=test_steps_per_epoch,
        verbose=1,
        epochs=EPOCHS
    )