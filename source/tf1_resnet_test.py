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

model = tf.keras.models.load_model('/home/barcelona/pervinco/source/cam/Grad-CAM-tensorflow/train/face_gender_glass.h5')
data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

test_generator = data_generator.flow_from_directory(
    directory='/home/barcelona/pervinco/datasets/face_gender_glass/predict',
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


TEST_DIR = '/home/barcelona/pervinco/datasets/face_gender_glass/predict/'
class_label = glob.glob('/home/barcelona/pervinco/datasets/face_gender_glass/train/*')
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

for i in range(0, 6):
    print(test_generator.filenames[i])
    img = cv2.imread(TEST_DIR + test_generator.filenames[i])
    predicted_class = labels[predicted_class_indices[i]]
    print(predicted_class)




