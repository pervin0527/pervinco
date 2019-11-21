from __future__ import absolute_import, division, print_function, unicode_literals
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
'''
definition datas.
train data = cats(1000) + dogs(1000)
validation data = cats(500) + dogs(500)
'''
dataset_dir = '/home/barcelona/pervinco/datasets/cats_and_dogs_filtered'
train_dir = os.path.join(dataset_dir, 'train')
validation_dir = os.path.join(dataset_dir, 'validation')

train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

train_data_num = len(os.listdir(train_dogs_dir) + os.listdir(train_cats_dir))
validation_data_num = len(os.listdir(validation_cats_dir) + os.listdir(validation_dogs_dir))

print('train_data num :', train_data_num)
print('validation_data num : ', validation_data_num)

'''
definition layers
'''
IMG_HEIGHT = 150
IMG_WIDTH = 150

model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation='relu'),
    #Dense(1, activation='sigmoid')
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam',
              # loss='binary_crossentropy'
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

'''
data preprocessing
1. read images from disk
2. convert jpeg to rgb
3. convert float type tensor
4. resize pixel values 0 ~ 255 ==> [0, 1]
'''
train_data_gen = ImageDataGenerator(rescale=1./255)
validation_data_gen = ImageDataGenerator(rescale=1./255)

train_generator = train_data_gen.flow_from_directory(
    train_dir,
    # resize train data
    target_size=(150, 150),
    batch_size=20,
    #binary crossentropy loss func
    class_mode='binary'
)

validation_generator = validation_data_gen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)

for data_batch, labels_batch in train_generator:
    print('batch data size : ', data_batch.shape)
    print('batch label size : ', labels_batch.shape)
    break

history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=50
)

model.save('/home/barcelona/pervinco/model/cats_and_dogs.h5')
