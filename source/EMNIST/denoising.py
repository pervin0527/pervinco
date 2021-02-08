import cv2, sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
from matplotlib import pyplot as plt
import albumentations as A

transforms = A.Compose([
                A.Resize(10, 10, p=1),
                # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1),
                # A.MedianBlur(blur_limit=3, always_apply=True, p=0.4),
                # A.HorizontalFlip(p=0.4),
                # A.VerticalFlip(p=0.4),
                # A.RandomRotate90(p=0.5),
            ])


"""salt pepper noise filter"""
image_path = '/data/backup/pervinco/datasets/dirty_mnist_2/dirty_mnist_2nd/00000.png'

image = cv2.imread(image_path) # cv2.IMREAD_GRAYSCALE
cv2.imshow('original', image)
cv2.waitKey(0)

image2 = np.where((image <= 254) & (image != 0), 0., image)
cv2.imshow('filterd', image2)
cv2.waitKey(0)

# kernel = np.ones((1, 1), np.uint8)
# image3 = cv2.erode(image2, kernel=kernel, iterations=1)
# cv2.imshow('erode', image3)
# cv2.waitKey(0)

# kernel2 = np.ones((2, 2), np.uint8)
# image4 = cv2.dilate(image3, kernel=kernel2, iterations=1)
# cv2.imshow('dilate', image4)
# cv2.waitKey(0)

augmented_image =transforms(image=image2)['image']
print(augmented_image)