import tensorflow as tf
from PIL import Image
from matplotlib.pyplot import imshow
import numpy as np
import cv2

img_path = "/data/backup/pervinco_2020/datasets/landmark_classification/4.19학생혁명기념탑/4.19학생혁명기념탑_001.JPG"
image = tf.io.read_file(img_path)
image = tf.image.decode_jpeg(image, channels=3)
image = tf.image.resize(image, [270, 480])

img = tf.keras.preprocessing.image.array_to_img(image)
img = cv2.cvtColor(np.asarray(img), cv2.COLOR_BGR2RGB)
cv2.imshow('test', img)
cv2.waitKey(0)