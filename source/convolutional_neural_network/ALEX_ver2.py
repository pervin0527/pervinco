from PIL import Image
import tensorflow as tf
from tensorflow import keras
from keras.utils import normalize, to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.datasets import cifar10

import numpy as np
import glob
import datetime
import pathlib

IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 128
epochs = 2000


def make_np(data_list, class_label):
    labels = []
    images = []
    for img in data_list:
        label = img.split('/')[-2]

        if label in class_label:
            labels.append(class_label.index(label))

        img = Image.open(img)
        img = img.resize((IMG_HEIGHT, IMG_WIDTH))
        img = np.array(img)
        images.append(img)

    np_images = np.array(images)
    np_labels = np.array(labels)
    np_labels = np_labels.reshape(len(labels), 1)
    # print('image np : ', np_images.shape)
    # print('label np : ', np_labels.shape)

    return np_images, np_labels


def data_to_np(train_dir, valid_dir):
    class_label = []

    train_data_list = glob.glob(str(train_dir) + '/*/*.jpg')
    valid_data_list = glob.glob(str(valid_dir) + '/*/*.jpg')
    train_label_list = glob.glob(str(train_dir) + '/*')

    for i in range(0, len(train_label_list)):
        class_label.append(train_label_list[i].split('/')[-1])

    print('Dataset Label : ', class_label)

    train_images, train_labels = make_np(train_data_list, class_label)
    valid_images, valid_labels = make_np(valid_data_list, class_label)

    return (train_images, train_labels), (valid_images, valid_labels)


def ALEX_NET():
    model = tf.keras.models.Sequential([
        # layer 1
        tf.keras.layers.Conv2D(filters=96,
                               kernel_size=(11, 11),
                               strides=4,
                               padding="same",
                               input_shape=(224, 224, 3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),

        tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                  strides=2,
                                  padding="same"),

        # layer 2
        tf.keras.layers.Conv2D(filters=256,
                               kernel_size=(5, 5),
                               strides=1,
                               padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                  strides=2,
                                  padding="same"),

        # layer 3
        tf.keras.layers.Conv2D(filters=384,
                               kernel_size=(3, 3),
                               strides=1,
                               padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),

        # layer 4
        tf.keras.layers.Conv2D(filters=384,
                               kernel_size=(3, 3),
                               strides=1,
                               padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),

        # layer 5
        tf.keras.layers.Conv2D(filters=256,
                               kernel_size=(3, 3),
                               strides=1,
                               padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),

        tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                  strides=2,
                                  padding="same"),
        # layer 6
        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(units=4096),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dropout(rate=0.4),

        # layer 7
        tf.keras.layers.Dense(units=4096),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dropout(rate=0.4),

        # layer 8
        tf.keras.layers.Dense(units=2,
                              activation="softmax")
    ])
    model.summary()
    return model


if __name__ == '__main__':
    train_dir = pathlib.Path('/home/barcelona/pervinco/datasets/cats_and_dogs_filtered/train')
    total_train_data = len(list(train_dir.glob('*/*.jpg')))
    print('total train data : ', total_train_data)

    valid_dir = pathlib.Path('/home/barcelona/pervinco/datasets/cats_and_dogs_filtered/validation')
    total_valid_Data = len(list(valid_dir.glob('*/*.jpg')))
    print('total validation data : ', total_valid_Data)

    (x_train, y_train), (x_test, y_test) = data_to_np(train_dir, valid_dir)
    # (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    print('train images, labels', x_train.shape, y_train.shape)
    print('validation images, labels', x_test.shape, y_test.shape)

    x_train = normalize(x_train, axis=1)
    y_train = to_categorical(y_train)
    x_test = normalize(x_test, axis=1)
    y_test = to_categorical(y_test)

    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)

    model = ALEX_NET()
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, decay=5e-5, momentum=0.9)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    train_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=45,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.5,
        featurewise_center=True,
        featurewise_std_normalization=True
        # shear_range=0.2
    )

    model.fit_generator(train_image_generator.flow(x_train, y_train, batch_size=BATCH_SIZE),
                        steps_per_epoch=len(x_train) / BATCH_SIZE, epochs=epochs,
                        callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=150, verbose=1)],
                        verbose=1)
