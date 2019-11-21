# -*- coding: utf-8 -*-
'''
https://www.tensorflow.org/tutorials/images/classification#load_data
https://www.tensorflow.org/tutorials/keras/save_and_load#saved_model%EC%9D%84_%EC%82%AC%EC%9A%A9%ED%95%98%EA%B8%B0
https://www.tensorflow.org/guide/keras/train_and_evaluate
'''
# python2를 python3처럼 사용할 수 있도록 만들어 준다.
from __future__ import absolute_import, division, print_function, unicode_literals

# import
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

'''
The os package is used to read files and directory structure
NumPy is used to convert python list to numpy array and to perform required matrix operations
matplotlib.pyplot to plot the graph and display images in the training and validation data.
'''
import numpy as np
import os
import matplotlib.pyplot as plt

train_dir = os.path.join('/home/barcelona/gg/datasets/cats_and_dogs_filtered/train')
val_dir = os.path.join('/home/barcelona/gg/datasets/cats_and_dogs_filtered/validation')

train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')

test_cats_dir = os.path.join(val_dir, 'cats')
test_dogs_dir = os.path.join(val_dir, 'dogs')

print(train_cats_dir)
print(train_dogs_dir)
print(test_cats_dir)
print(test_dogs_dir)

num_cats_tr = len(os.listdir(train_cats_dir))
num_dogs_tr = len(os.listdir(train_dogs_dir))
num_cats_te = len(os.listdir(test_cats_dir))
num_dogs_te = len(os.listdir(test_dogs_dir))

total_train = num_cats_tr + num_dogs_tr
total_val = num_cats_te + num_dogs_te

print(total_train)
print(total_val)

batch_size = 200
epochs = 30
IMG_HEIGHT = 150
IMG_WIDTH = 150

'''
class ImageDataGenerator : generate batches of tensor image data with real-time data augmentation
flow_from_directory : Takes the path to a directory & generates batches of augmented data
flow_from_directory method load images from the disk, applies rescaling, and resizes the images
'''
train_image_generator = ImageDataGenerator(rescale=1./255,
                                           rotation_range=45,
                                           width_shift_range=.15,
                                           height_shift_range=.15,
                                           horizontal_flip=True,
                                           zoom_range=0.5)

validation_image_generator = ImageDataGenerator(rescale=1./255)

train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='binary')

val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=val_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='binary')

'''
next를 사용하는 이유는 batch되는 데이터를 sample로 보여주기 위해서 사
next function returns a batch from the dataset.
The return value of next function is in form of (x_train = features, y_train = labels)
next는 iterator에서 다음 순번으로 넘어가게 하는 것.
train_data_gen에는 train data가 batch size만큼 들어 있고, 그것을 순차적으로 넘어가기 위해 next를 사용
'''
training_images, training_labels = next(train_data_gen)


def plotImages(images_arr):
    fig, axes = plt.subplots(1, 4, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


# plotImages(training_images[:5]) ##128개 중 5개를 샘플로 보여줌.
# print(training_images) ## feature 값. pixel 값에 255로 나눈 값.
# print(training_labels) ## batch size만큼 나옴

model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu',
           input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    MaxPooling2D(),
    Dropout(0.2),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Dropout(0.2),
    Flatten(),
    Dense(512, activation='relu'),
    # softmax 2개의 확률을 반환하고 반환된 값의 전체 합은 1.
    # 각 노드는 현재 이미지가 2개 클래스 중 하나에 속할 확률을 출력.
    # sigmoid 0에서 1사이 출력 값을 반환. 정리하면 class가 2일 경우 softmax = sigmoid
    # Dense(2, activation='softmax')
    Dense(1, activation='sigmoid')
])

# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

checkpoint_path = "/home/barcelona/gg/model/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
model.save_weights(checkpoint_path.format(epoch=0))


'''
python generator에서 배치별로 생성된 데이터를 모델에 적용
model.fit은 설정한 batch size만큼 input하고 epoch 동안 반복.
model.fit_generator는 python의 generator에 의해 batch size만큼 input하고 다음 데이터들로 batch
'''
history = model.fit_generator(
    train_data_gen,
    steps_per_epoch = total_train // batch_size,
    epochs = epochs,
    validation_data = val_data_gen,
    validation_steps = total_val // batch_size,
    callbacks = [cp_callback]
)

# val_loss, val_acc = model.evaluate_generator(generator=valid_data_gen, steps=1, verbose=2)
# print('valid accuracy : ', val_acc)


acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
