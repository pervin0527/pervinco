import cv2, sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
from matplotlib import pyplot as plt
import albumentations as A
import tensorflow as tf

transforms = A.Compose([A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1),])

"""salt pepper noise filter"""
image_path = '/data/backup/pervinco/datasets/dirty_mnist_2/test_dirty_mnist_2nd/50001.png'
images = []

image = cv2.imread(image_path) # cv2.IMREAD_GRAYSCALE
images.append(('original', image))

image2 = np.where((image <= 254) & (image != 0), 0, image)
images.append(('filtered', image2))

image3 = cv2.dilate(image2, kernel=np.ones((2, 2), np.uint8), iterations=1)
images.append(('dilate', image3))

image4 = cv2.medianBlur(image3, 5)
images.append(('median', image4))

image5 = image4 - image2
images.append(('sub', image5))

image6 = cv2.erode(image2, kernel=np.ones((2, 2), np.uint8), iterations=1)
image6 = cv2.dilate(image6, kernel=np.ones((3, 3), np.uint8), iterations=1)
images.append(('test', image6))

fig = plt.figure()

idx = 1
for title, image in images:
    ax = fig.add_subplot(1, len(images), idx)
    ax.imshow(image)
    ax.set_xlabel(title)
    ax.set_xticks([]), ax.set_yticks([])
    idx += 1

plt.show()

alb_test = transforms(image=image5)['image']
print(alb_test[0][:5])

tf_test = tf.keras.applications.resnet.preprocess_input(image5)
print(tf_test[0][:5])