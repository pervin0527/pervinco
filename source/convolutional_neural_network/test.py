import tensorflow as tf
import sys, os
import argparse
import datetime
# import tkinter
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import csv
from PIL import Image

IMG_HEIGHT = 227
IMG_WIDTH = 227
BATCH_SIZE = 128
EPOCHS = 10

name = sys.argv[1]

def write_csv(pixel_value):
    with open('./result_csv/' + name + '_metrics.csv', 'a') as df:
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
    plt.savefig('/home/tae/ssd_300/source/convolutional_neural_network/feature_map'
                + '/' + name + '/' + title)


def model():
    inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))

    conv1 = tf.keras.layers.Conv2D(filters=96, kernel_size=(11, 11), strides=4, padding='valid',
                                input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
                                #activation='relu',
                                use_bias=False)(inputs)
    norm1 = tf.keras.layers.BatchNormalization()(conv1)
    activation1 = tf.keras.layers.ReLU()(norm1)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2)(activation1)

    conv2 = tf.keras.layers.Conv2D(filters=256, kernel_size=(5, 5), 
                                   padding='same',
                                   #activation='relu',
                                   use_bias=False)(pool1)
    norm2 = tf.keras.layers.BatchNormalization()(conv2)
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


if __name__ == '__main__':

    x_train = '/home/tae/ssd_300/datasets/predict/cat_dog/cat/google_0000.jpg'
    x_train = Image.open(x_train)
    x_train = x_train.resize((IMG_HEIGHT, IMG_WIDTH))
    x_train = np.array(x_train)
    x_train = x_train.astype('float32') / 255

    model = model()
    model.summary()

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    img = np.reshape(x_train, (-1, IMG_HEIGHT, IMG_WIDTH, 3))
    inputs = model.layers[0].output

    conv1 = model.layers[1].output

    conv2 = model.layers[2].output
    norm2 = model.layers[3].output
    pool2 = model.layers[4].output

    conv3 = model.layers[5].output
    norm3 = model.layers[6].output
    pool3 = model.layers[7].output

    conv4 = model.layers[8].output
    conv5 = model.layers[9].output
    pool5 = model.layers[10].output

    print('input shape : ', inputs.shape, inputs.dtype)

    print('conv1 shape : ', conv1.shape)
    model = tf.keras.models.Model(inputs=inputs, outputs=conv1)
    output = model.predict(img)
    print('conv1 output shape', output.shape)
    show_layer_output(output, '1_conv1')

    print('conv2 shape', conv2.shape)
    model = tf.keras.models.Model(inputs=inputs, outputs=conv2)
    output = model.predict(img)
    print('conv2 output shape', output.shape)
    write_csv(output[0,0,0])
    show_layer_output(output, '2_conv2')

    print('norm2 shape :', norm2.shape)
    model = tf.keras.models.Model(inputs=inputs, outputs=norm2)
    output = model.predict(img)
    print('norm2 output shape', output.shape)
    write_csv(output[0,0,0])
    show_layer_output(output, '2_norm2')

    print('pool2 shape :', pool2.shape)
    model = tf.keras.models.Model(inputs=inputs, outputs=pool2)
    output = model.predict(img)
    print('pool2 output shape', output.shape)
    write_csv(output[0,0,0])
    show_layer_output(output, '2_pool2')

    print('conv3 shape : ', conv3.shape)
    model = tf.keras.models.Model(inputs=inputs, outputs=conv3)
    output = model.predict(img)
    print('conv3 output shape', output.shape)
    write_csv(output[0,0,0])
    show_layer_output(output, '3_conv3')

    print('norm3 shape : ', norm3.shape)
    model = tf.keras.models.Model(inputs=inputs, outputs=norm3)
    output = model.predict(img)
    print('norm3 output shape', output.shape)
    write_csv(output[0,0,0])
    show_layer_output(output, '3_norm3')

    print('pool3 shape : ', pool3.shape)
    model = tf.keras.models.Model(inputs=inputs, outputs=pool3)
    output = model.predict(img)
    print('pool3 output shape', output.shape)
    write_csv(output[0,0,0])
    show_layer_output(output, '3_pool3')

    print('conv4 shape : ', conv4.shape)
    model = tf.keras.models.Model(inputs=inputs, outputs=conv4)
    output = model.predict(img)
    print('conv4 output shape', output.shape)
    write_csv(output[0,0,0])
    show_layer_output(output, '4_conv4')

    print('conv5 shape : ', conv5.shape)
    model = tf.keras.models.Model(inputs=inputs, outputs=conv5)
    output = model.predict(img)
    print('conv5 output shape', output.shape)
    write_csv(output[0,0,0])
    show_layer_output(output, '5_conv5')

    print('pool5 shape : ', pool3.shape)
    model = tf.keras.models.Model(inputs=inputs, outputs=pool5)
    output = model.predict(img)
    print('pool5 output shape', output.shape)
    write_csv(output[0,0,0])
    show_layer_output(output, '5_pool5')
