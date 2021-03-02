import tensorflow as tf
from PIL import Image
from matplotlib.pyplot import imshow
import numpy as np
import cv2

img_path = "/data/backup/pervinco/datasets/plant-pathology-2020-fgvc7/images/Test_3.jpg"
print(f"Load Image : {img_path}")
image = tf.io.read_file(img_path)
image = tf.image.decode_jpeg(image, channels=3)
image = tf.image.resize(image, [270, 480])

img = tf.keras.preprocessing.image.array_to_img(image)
img = cv2.cvtColor(np.asarray(img), cv2.COLOR_BGR2RGB)
cv2.imshow('test', img)
cv2.waitKey(0)