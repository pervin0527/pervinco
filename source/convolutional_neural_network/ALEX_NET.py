'''
            ---------------------------------update 19.12.03 --------------------------------------
            training model : ALEX_NET
            epochs 380/1200
            train_loss : 0.06839 train_acc : 97.44
            valid_loss : 0.3677 valid_acc : 90.18
            --------------------------------update 19.12.06----------------------------------------
            training model : ALEX_NET
            training data inceresed 3900
            train_loss: 0.0475 - train_accuracy: 0.9828
            val_loss: 0.3382 - val_accuracy: 0.9040
            ---------------------------------------------------------------------------------------
'''

from __future__ import absolute_import, division, print_function, unicode_literals
import os
import cv2
import tensorflow as tf
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import datetime
from tensorflow import keras


tf.executing_eagerly()
AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 128
IMG_HEIGHT = 224
IMG_WIDTH = 224
epochs = 1200

train_dir = pathlib.Path('/home/barcelona/pervinco/datasets/cats_and_dogs_filtered/train')
valid_dir = pathlib.Path('/home/barcelona/pervinco/datasets/cats_and_dogs_filtered/validation')
total_train_data = len(list(train_dir.glob('*/*.jpg')))
total_val_data = len(list(valid_dir.glob('*/*.jpg')))
print(total_train_data)
CLASS_NAMES = np.array([item.name for item in train_dir.glob('*') if item.name != "LICENSE.txt"])

def ALEX_NET():
    inputs = keras.Input(shape=(224, 224, 3))

    conv1 = keras.layers.Conv2D(filters=96, kernel_size=(11, 11), strides=4, padding='same',
                                input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
                                activation='relu')(inputs)

    conv2 = keras.layers.Conv2D(filters=256, kernel_size=(5, 5), padding='same', activation='relu')(conv1)
    norm1 = tf.nn.local_response_normalization(conv2)
    pool1 = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2)(norm1)

    conv3 = keras.layers.Conv2D(filters=384, kernel_size=(3, 3), padding='same', activation='relu')(pool1)
    norm2 = tf.nn.local_response_normalization(conv3)
    pool2 = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2)(norm2)

    conv4 = keras.layers.Conv2D(filters=384, kernel_size=(3, 3), padding='same', activation='relu')(pool2)
    conv5 = keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(conv4)
    pool3 = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2)(conv5)

    flat = keras.layers.Flatten()(pool3)
    dense1 = keras.layers.Dense(4096, activation='relu')(flat)
    drop1 = keras.layers.Dropout(0.5)(dense1)
    dense2 = keras.layers.Dense(4096, activation='relu')(drop1)
    drop2 = keras.layers.Dropout(0.5)(dense2)
    dense3 = keras.layers.Dense(2, activation='softmax')(drop2)
    return keras.Model(inputs=inputs, outputs=dense3)


model = ALEX_NET()
model.summary()


optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, decay=5e-5, momentum=0.9)
model.compile(
    # optimizer='adam',
    optimizer=optimizer,
    # loss='binary_crossentropy'
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

train_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
                                                                        rescale=1./255,
                                                                        rotation_range=45,
                                                                        width_shift_range=.15,
                                                                        height_shift_range=.15,
                                                                        horizontal_flip=True,
                                                                        zoom_range=0.5
                                                                        # shear_range=0.2
)

valid_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_generator = train_image_generator.flow_from_directory(
    directory=train_dir,
    # resize train data
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    shuffle=True,
    class_mode='categorical',
)

valid_generator = valid_image_generator.flow_from_directory(
    directory=valid_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

start_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

log_dir = '/home/barcelona/pervinco/model/logs' + start_time
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
early_stopping_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=150, verbose=1)
history = model.fit_generator(
    train_generator,
    steps_per_epoch=total_train_data//BATCH_SIZE,
    epochs=epochs,
    verbose=1,
    validation_data=valid_generator,
    validation_steps=total_val_data//BATCH_SIZE,
    callbacks=[early_stopping_callback, tensorboard_callback]
)

model.save('/home/barcelona/pervinco/model/'+start_time+'.h5')
