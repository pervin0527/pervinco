import string, cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow import keras
from tensorflow.keras import layers

(images1, labels1), (images2, labels2) = tfds.as_numpy(tfds.load('emnist/letters', split=['train', 'test'], batch_size=-1, as_supervised=True,))

images1 = images1 / 255
images2 = images2 / 255

labels1 -= 1
labels2 -= 1

# csv = pd.read_csv('/data/backup/pervinco/datasets/dirty_mnist_2/mnist_data_2nd/train.csv')
# images3 = csv.drop(['id', 'digit', 'letter'], axis=1).values
# images= images3 / 255
# images3 = images3.reshape(-1, 28, 28, 1)
# images3 = np.where((images3 <= 20) & (images3 != 0), 0., images3)

# alphabets = list(string.ascii_uppercase)
# labels3 = list(csv['letter'])

# for idx, value in enumerate(labels3):
#     labels3[idx] = alphabets.index(value)

# train_images = np.concatenate((images1, images3))
# train_labels = np.concatenate((labels1, labels3))

train_images = images1
train_labels = labels1
test_images = images2
test_labels = labels2

noise = 0.5
noisy_train_images = train_images + noise * np.random.normal(0, 1, size=train_images.shape)
noisy_test_images = test_images + noise * np.random.normal(0, 1, size=test_images.shape)

noisy_train_images = np.clip(noisy_train_images, 0, 1)
noisy_test_images = np.clip(noisy_test_images, 0, 1)

inputs = tf.keras.Input(shape=(28,28,1))

x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
x = tf.keras.layers.MaxPool2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(x)
encoded = tf.keras.layers.MaxPool2D()(x)

x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(encoded)
x = tf.keras.layers.UpSampling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(x)
x = tf.keras.layers.UpSampling2D()(x)
decoded = tf.keras.layers.Conv2D(1, 3, activation='sigmoid', padding='same')(x)

autoencoder = tf.keras.Model(inputs, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.summary()

epochs = 20
batch_size = 256

history = autoencoder.fit(noisy_train_images,
                          train_images,
                          epochs=epochs,
                          batch_size=batch_size,
                          shuffle=True,
                          validation_data=(noisy_test_images, test_images))

autoencoder.save('/data/backup/pervinco/test_code/autoencoder.h5')


# test_images = noisy_test_images[rand:rand+num_imgs]
# test_images = tf.keras.preprocessing.image.load_img('/data/backup/pervinco/datasets/dirty_mnist_2/test_dirty_mnist_2nd/50000.png', color_mode='grayscale', target_size=(28,28))
# test_images = tf.keras.preprocessing.image.img_to_array(test_images)
# test_images = np.array([test_images])
# print(test_images.shape)

# autoencoder = tf.keras.models.load_model('/data/backup/pervinco/test_code/autoencoder.h5')
# test_desoided = autoencoder.predict(test_images)
# print(test_desoided.shape)
# print(test_desoided[0].shape)

# cv2.imshow('result', test_desoided[0])
# cv2.waitKey(0)