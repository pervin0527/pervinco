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
    inputs = keras.Input(shape=(224, 224, 3))

    conv1 = keras.layers.Conv2D(filters=96, kernel_size=(11, 11), strides=4, padding='same',
                                input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
                                activation='relu')(inputs)

    conv2 = keras.layers.Conv2D(filters=256, kernel_size=(5, 5), padding='same', kernel_initializer='he_uniform',
                                activation='relu')(conv1)
    norm1 = tf.nn.local_response_normalization(conv2)
    # norm1 = keras.layers.BatchNormalization()(conv2)
    pool1 = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2)(norm1)

    conv3 = keras.layers.Conv2D(filters=384, kernel_size=(3, 3), padding='same', kernel_initializer='he_uniform',
                                activation='relu')(pool1)
    norm2 = tf.nn.local_response_normalization(conv3)
    # norm2 = keras.layers.BatchNormalization()(conv3)
    pool2 = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2)(norm2)

    conv4 = keras.layers.Conv2D(filters=384, kernel_size=(3, 3), padding='same', kernel_initializer='he_uniform',
                                activation='relu')(pool2)
    conv5 = keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', kernel_initializer='he_uniform',
                                activation='relu')(conv4)
    pool3 = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2)(conv5)

    flat = keras.layers.Flatten()(pool3)
    dense1 = keras.layers.Dense(4096, activation='relu', kernel_initializer='he_uniform')(flat)
    drop1 = keras.layers.Dropout(0.5)(dense1)
    dense2 = keras.layers.Dense(4096, activation='relu', kernel_initializer='he_uniform')(drop1)
    drop2 = keras.layers.Dropout(0.5)(dense2)
    dense3 = keras.layers.Dense(2, activation='softmax')(drop2)
    return keras.Model(inputs=inputs, outputs=dense3)


if __name__ == '__main__':
    train_dir = pathlib.Path('/home/tae/data/pervinco/datasets/cats_and_dogs_filtered/train')
    total_train_data = len(list(train_dir.glob('*/*.jpg')))
    print('total train data : ', total_train_data)

    valid_dir = pathlib.Path('/home/tae/data/pervinco/datasets/cats_and_dogs_filtered/validation')
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

    train_image_generator = ImageDataGenerator(rotation_range=45,
                                               width_shift_range=.15,
                                               height_shift_range=.15,
                                               horizontal_flip=True,
                                               zoom_range=0.5,
                                               # shear_range=0.2
                                               )

    train_generator = train_image_generator.flow(x_train, y_train, batch_size=BATCH_SIZE)
    model = ALEX_NET()
    model.summary()

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, decay=5e-5, momentum=0.9)
    model.compile(
        # optimizer='rmsprop',
        optimizer=optimizer,
        # loss='binary_crossentropy'
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    log_dir = '/home/tae/data/pervinco/model/logs' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=total_train_data // BATCH_SIZE,
        epochs=epochs,
        shuffle=True,
        verbose=1,
        validation_data=(x_test, y_test),
        callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=100, verbose=1),
                   tensorboard_callback]
    )