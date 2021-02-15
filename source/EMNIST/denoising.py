import cv2, sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
from matplotlib import pyplot as plt
import albumentations as A
import tensorflow as tf

transforms = A.Compose([
                # A.Resize(10, 10, p=1),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1),
                # A.MedianBlur(blur_limit=3, always_apply=True, p=0.4),
                # A.HorizontalFlip(p=0.4),
                # A.VerticalFlip(p=0.4),
                # A.RandomRotate90(p=0.5),
            ])


"""salt pepper noise filter"""
image_path = '/data/backup/pervinco/datasets/dirty_mnist_2/dirty_mnist_2nd/00018.png'

image = cv2.imread(image_path) # cv2.IMREAD_GRAYSCALE
cv2.imshow('original', image)

image2 = np.where((image <= 254) & (image != 0), 0, image)
cv2.imshow('filterd', image2)

image3 = cv2.dilate(image2, kernel=np.ones((2, 2), np.uint8), iterations=1)
cv2.imshow('dilate', image3)


image4 = cv2.medianBlur(image3, 5)
cv2.imshow('median', image4)

image5 = image4 - image2
cv2.imshow('sub', image5)

cv2.waitKey(0)

alb_test = transforms(image=image5)['image']
print(alb_test[0][:5])

tf_test = tf.keras.applications.resnet.preprocess_input(image5)
print(tf_test[0][:5])