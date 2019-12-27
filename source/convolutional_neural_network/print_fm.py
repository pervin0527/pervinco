import tensorflow as tf
import datetime
# import tkinter
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import csv
import argparse
import sys, os
import matplotlib
from PIL import Image
np.seterr(divide='ignore', invalid='ignore')
np.set_printoptions(threshold=sys.maxsize)


IMG_HEIGHT = 227
IMG_WIDTH = 227
BATCH_SIZE = 128
EPOCHS = 10

name = sys.argv[1]
layer_num = int(sys.argv[2])


def write_csv(pixel_value):
    with open('./csv/' + name + '_metrics.csv', 'a') as df:
        write = csv.writer(df, delimiter=',')
        write.writerow([pixel_value])


def show_layer_output(output, title):
    plt.figure(figsize=(20, 20))  # window size
    size = 0
    for i in range(output.shape[-1]):
        size += 1
    print(size)

    for i in range(output.shape[-1]):  # i value = (0 ~ channels)
        plt.subplot(8, size/8, i + 1)
        plt.axis('off')
        plt.matshow(output[0, :, :, i], cmap='gray', fignum=0)
    plt.tight_layout()
    plt.savefig('/home/barcelona/pervinco/source/convolutional_neural_network/feature_map/' + title)


def model():
    inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))

    conv1 = tf.keras.layers.Conv2D(filters=96, kernel_size=(11, 11), strides=4, padding='valid',
                                   input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))(inputs)
    norm1 = tf.keras.layers.BatchNormalization(axis=-1, center=True)(conv1)
    activation1 = tf.keras.layers.ReLU()(norm1)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2)(activation1)

    conv2 = tf.keras.layers.Conv2D(filters=256, kernel_size=(5, 5),
                                   padding='same')(pool1)
    norm2 = tf.keras.layers.BatchNormalization(axis=-1, center=True)(conv2)
    activation2 = tf.keras.layers.ReLU()(norm2)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2)(activation2)

    conv3 = tf.keras.layers.Conv2D(filters=384, kernel_size=(3, 3),
                                   padding='same', activation='relu')(pool2)
    conv4 = tf.keras.layers.Conv2D(filters=384, kernel_size=(3, 3),
                                   padding='same', activation='relu')(conv3)
    conv5 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3),
                                   padding='same', activation='relu')(conv4)

    pool5 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2)(conv5)

    flat = tf.keras.layers.Flatten()(pool5)
    dense1 = tf.keras.layers.Dense(4096, activation='relu')(flat)
    drop1 = tf.keras.layers.Dropout(0.5)(dense1)
    dense2 = tf.keras.layers.Dense(4096, activation='relu')(drop1)
    drop2 = tf.keras.layers.Dropout(0.5)(dense2)
    outputs = tf.keras.layers.Dense(2, activation='softmax')(drop2)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def load_image(test_path, target_size):
    img = tf.keras.preprocessing.image.load_img(test_path, target_size=target_size)
    img_tensor=tf.keras.preprocessing.image.img_to_array(img)

    return img_tensor[np.newaxis]/255


def predict_and_get_outputs(model, img_path):
    layer_outputs = [layer.output for layer in model.layers[1:layer_num]]
    layer_names = [layer.name for layer in model.layers[1:layer_num]]

    print([str(output.shape) for output in layer_outputs])
    activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)

    input_shape = (model.input.shape[1], model.input.shape[2])      # (150, 150)
    img_tensor = load_image(img_path, target_size=input_shape)

    for layer in layer_outputs:
        print(layer.shape)
    print('-' * 50)

    layer_outputs = activation_model.predict(img_tensor)
    for layer in layer_outputs:
        print(layer.shape)
        write_csv(layer[0, 0, 0])

    return layer_outputs, layer_names


def show_activation_maps(layer, title, layer_index, n_cols=16):
    size, n_features = layer.shape[1], layer.shape[-1]
    assert n_features % n_cols == 0

    n_rows = n_features // n_cols

    big_image = np.zeros((n_rows*size, n_cols*size), dtype=np.float32)

    for row in range(n_rows):
        for col in range(n_cols):
            channel = layer[0, :, :, row * n_cols + col]      # shape : (size, size)

            channel -= channel.mean()
            channel /= channel.std()
            channel *= 64
            channel += 128
            channel = np.clip(channel, 0, 255).astype('uint8')

            big_image[row*size:(row+1)*size, col*size:(col+1)*size] = channel

    plt.figure(figsize=(n_cols, n_rows))

    plt.xticks(np.arange(n_cols) * size)
    plt.yticks(np.arange(n_rows) * size)
    plt.title('layer {} : {}'.format(layer_index, title))
    plt.tight_layout()
    plt.imshow(big_image, cmap='gray')           # cmap='gray'


if __name__ == '__main__':

    test_path = '/home/barcelona/pervinco/datasets/predict/cat_dog/dog/google_0002.jpg'

    model = model()
    model.summary()

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    layer_outputs, layer_names = predict_and_get_outputs(model, test_path)

    for i, (layer, name) in enumerate(zip(layer_outputs, layer_names)):
        show_activation_maps(layer, name, i)

    plt.show()
