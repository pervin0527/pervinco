'''
2019.12.30 : loss: 0.0841 - accuracy: 0.9655 - val_loss: 0.1092 - val_accuracy: 0.9576
'''

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import pathlib
import numpy as np
import datetime
from tensorflow import keras

tf.executing_eagerly()
AUTOTUNE = tf.data.experimental.AUTOTUNE

BATCH_SIZE = 128
IMG_HEIGHT = 227
IMG_WIDTH = 227
epochs = 2000

train_dir = pathlib.Path('/home/barcelona/pervinco/datasets/cats_and_dogs_filtered/train')
valid_dir = pathlib.Path('/home/barcelona/pervinco/datasets/cats_and_dogs_filtered/validation')
total_train_data = len(list(train_dir.glob('*/*.jpg')))
total_val_data = len(list(valid_dir.glob('*/*.jpg')))
print(total_train_data)
CLASS_NAMES = np.array([item.name for item in train_dir.glob('*') if item.name != "LICENSE.txt"])

start_time = 'ALEX1_2class_'+ datetime.datetime.now().strftime("%Y.%m.%d_%H:%M:%S")
log_dir = '/home/barcelona/pervinco/model/' + start_time


def model():
    inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))

    conv1 = tf.keras.layers.Conv2D(filters=96, kernel_size=(11, 11), strides=4, padding='valid',
                                   activation='relu',
                                   input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))(inputs)
    norm1 = tf.nn.local_response_normalization(conv1)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2)(norm1)

    conv2 = tf.keras.layers.Conv2D(filters=256, kernel_size=(5, 5),
                                   padding='same', activation='relu')(pool1)
    norm2 = tf.nn.local_response_normalization(conv2)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2)(norm2)

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


model = model()
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

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
early_stopping_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, verbose=1)
history = model.fit_generator(
    train_generator,
    steps_per_epoch=total_train_data//BATCH_SIZE,
    epochs=epochs,
    verbose=1,
    validation_data=valid_generator,
    validation_steps=total_val_data//BATCH_SIZE,
    callbacks=[early_stopping_callback, tensorboard_callback]
    # callbacks=[tensorboard_callback]
)

model.save(log_dir+'/'+start_time+'.h5')