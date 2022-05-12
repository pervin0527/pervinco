import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import albumentations as A
import matplotlib.pyplot as plt

from model import DeepLabV3Plus
from glob import glob
from functools import partial
from tensorflow.keras import backend as K
from IPython.display import clear_output

# GPU setup
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


def decode_segmentation_masks(mask, colormap, n_classes):
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    for l in range(0, n_classes):
        idx = mask == l
        r[idx] = colormap[l, 0]
        g[idx] = colormap[l, 1]
        b[idx] = colormap[l, 2]
    rgb = np.stack([r, g, b], axis=2)

    return rgb


def visualize(display_list):
    fig = plt.figure(figsize=(6, 6))
    rows, cols = 1, 2

    x_labels = ["Train image", "Train mask"]
    for idx, image in enumerate(display_list):
        ax = fig.add_subplot(rows, cols, idx+1)
        # if image.shape[-1] == 3:
        #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax.imshow(image)
        ax.set_xlabel(x_labels[idx])
        ax.set_xticks([]), ax.set_yticks([])
    
    plt.show()


def get_file_list(path):
    images = sorted(glob(f"{path}/images/*.jpg"))
    masks = sorted(glob(f"{path}/masks/*.png"))
    n_images, n_masks = len(images), len(masks)
    
    return images, masks, n_images, n_masks


def load_data(image_path, mask_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, size=[IMG_SIZE, IMG_SIZE])

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, size=[IMG_SIZE, IMG_SIZE])

    return image, mask


# def one_hot_encoding(image, mask):
#     mask = tf.squeeze(mask, axis=-1)
#     mask = tf.cast(mask, dtype=tf.uint8)
#     mask = tf.one_hot(mask, len(CLASSES))

#     return image, mask


def build_dataset(images, masks, is_train):
    transform = A.Compose([
        # A.RandomSizedCrop(min_max_height=(IMG_SIZE/2, IMG_SIZE), height=IMG_SIZE, width=IMG_SIZE, p=0.5),
        A.OneOf([
            A.VerticalFlip(p=0.3),
            A.HorizontalFlip(p=0.3),
            A.Transpose(p=0.3)
        ], p=0.5),

        A.OneOf([
            A.ShiftScaleRotate(p=0.25, border_mode=0),
            A.RandomRotate90(p=0.25),
        ], p=1),

        A.OneOf([
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.5),
        ], p=1),

        A.OneOf([
            A.GridDropout(fill_value=0, mask_fill_value=0, random_offset=True, p=0.5),
            A.CoarseDropout(min_holes=1, max_holes=1,
                            min_height=80, min_width=80,
                            max_height=160, max_width=160,
                            fill_value=0, mask_fill_value=0,
                            p=0.5),
        ], p=0.6),
    ])

    def data_transform(image, mask):
        data = {"image" : image, "mask" : mask}
        transformed_data = transform(**data)
        transformed_image, transformed_mask = transformed_data["image"], transformed_data["mask"]
        transformed_image, transformed_mask = tf.cast(transformed_image, dtype=tf.uint8), tf.cast(transformed_mask, dtype=tf.uint8)

        return transformed_image, transformed_mask

    def process_data(image, mask):
        transformed_image, transformed_mask = tf.numpy_function(func=data_transform, inp=[image, mask], Tout=[tf.uint8, tf.uint8])

        return transformed_image, transformed_mask

    def set_shapes(image, mask):
        image.set_shape((IMG_SIZE, IMG_SIZE, 3))
        mask.set_shape((IMG_SIZE, IMG_SIZE, 1))

        return image, mask

    dataset = tf.data.Dataset.from_tensor_slices((images, masks))
    dataset = dataset.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)

    if is_train:
        dataset = dataset.map(partial(process_data), num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(set_shapes, num_parallel_calls=tf.data.AUTOTUNE)

    # if CATEGORICAL:
    #     dataset = dataset.map(one_hot_encoding, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.repeat()
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


if __name__ == "__main__":
    ROOT = "/data/Datasets/VOCdevkit/VOC2012"
    LABEL_PATH = f"{ROOT}/Labels/class_labels.txt"

    SAVE_PATH = "/data/Models/segmentation"    
    FOLDER = "SAMPLE05"

    CATEGORICAL = False
    BACKBONE_TRAINABLE = True
    BACKBONE_NAME = "ResNet50" # Xception, ResNet50, ResNet101
    FINAL_ACTIVATION = "softmax" # None, softmax
    # SAVE_NAME = f"{ROOT.split('/')[-1]}-{BACKBONE_NAME}-{FOLDER}-combined"
    SAVE_NAME = "TEST"

    BATCH_SIZE = 16
    EPOCHS = 300
    IMG_SIZE = 320
    LR_START = 0.00001

    label_df = pd.read_csv(LABEL_PATH, lineterminator='\n', header=None, index_col=False)
    CLASSES = label_df[0].to_list()
    print(CLASSES)

    COLORMAP = [[0, 0, 0], # background
                [128, 0, 0], # aeroplane
                [0, 128, 0], # bicycle
                [128, 128, 0], # bird
                [0, 0, 128], # boat
                [128, 0, 128], # bottle
                [0, 128, 128], # bus
                [128, 128, 128], # car
                [64, 0, 0], # cat
                [192, 0, 0], # chair
                [64, 128, 0], # cow
                [192, 128, 0], # diningtable
                [64, 0, 128], # dog
                [192, 0, 128], # horse
                [64, 128, 128], # motorbike
                [192, 128, 128], # person
                [0, 64, 0], # potted plant
                [128, 64, 0], # sheep
                [0, 192, 0], # sofa
                [128, 192, 0], # train
                [0, 64, 128] # tv/monitor
        ]
    COLORMAP = np.array(COLORMAP, dtype=np.uint8)

    train_dir = f"{ROOT}/{FOLDER}/train"
    valid_dir = f"{ROOT}/{FOLDER}/valid"
    train_images, train_masks, n_train_images, n_train_masks = get_file_list(train_dir)
    valid_images, valid_masks, n_valid_images, n_valid_masks = get_file_list(valid_dir)

    train_dataset = build_dataset(train_images, train_masks, True)
    valid_dataset = build_dataset(valid_images, valid_masks, False)

    print("Train Dataset:", train_dataset)
    print("Val Dataset:", valid_dataset)

    for item in train_dataset.take(1):
        image, mask = item[0][0], item[1][0]
        image = image.numpy()
        print(type(image))
        mask = decode_segmentation_masks(mask.numpy(), COLORMAP, len(CLASSES))
        mask = np.squeeze(mask, axis=-1)

        visualize([image, mask])