import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import tensorflow as tf
import glob
from tensorflow.python.keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.layers import Dense

IMAGE_RESIZE = 224
BATCH_SIZE_TESTING = 1
model_name = 'dog_cls'
dataset_name = model_name

saved_path = '/home/barcelona/pervinco/model/dog_cls/2020.02.03_14:49/'

test_img_dir = '/home/barcelona/pervinco/datasets/' + dataset_name + '/predict'
class_label = glob.glob('/home/barcelona/pervinco/datasets/' + dataset_name + '/train/*')

test_img_len = len(glob.glob(test_img_dir + '/test/*'))


model = tf.keras.models.load_model(saved_path + model_name + '.h5')
model.load_weights('/home/barcelona/pervinco/model/dog_cls/2020.02.03_14:49/08-0.93.hdf5')
data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

test_generator = data_generator.flow_from_directory(
    directory=test_img_dir,
    target_size=(IMAGE_RESIZE, IMAGE_RESIZE),
    batch_size=BATCH_SIZE_TESTING,
    class_mode=None,
    shuffle=False,
    seed=123
)


test_generator.reset()

pred = model.predict_generator(test_generator, steps=len(test_generator), verbose=1)
# print(pred)

predicted_class_indices = np.argmax(pred, axis=1)
# print(predicted_class_indices)


TEST_DIR = test_img_dir + '/'
# print(class_label)
labels = []

for i in class_label:
    i = i.split('/')[-1]
    labels.append(i)

labels = sorted(labels)    
print(labels)

# f, ax = plt.subplots(5, 5, figsize=(15, 15))
#
# for i in range(0, 5):
#     imgBGR = cv2.imread(TEST_DIR + test_generator.filenames[i])
#     imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)
#
#     # a if condition else b
#     predicted_class = labels[predicted_class_indices[i]]
#
#     ax[i//5, i % 5].imshow(imgRGB)
#     ax[i//5, i % 5].axis('off')
#     ax[i//5, i % 5].set_title("Predicted:{}".format(predicted_class))
#
# plt.show()

for i in range(0, test_img_len):
    img = cv2.imread(TEST_DIR + test_generator.filenames[i])
    predicted_class = labels[predicted_class_indices[i]]
    print(test_generator.filenames[i], 'predict result : ', predicted_class)




