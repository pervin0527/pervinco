# -*- coding: utf-8 -*-
import numpy as np
import time
import cv2
from albumentations import (
    ShiftScaleRotate, CLAHE, Blur, GridDistortion, IAAAdditiveGaussianNoise, GaussNoise, RandomSizedCrop, Resize,
    MotionBlur, MedianBlur, IAASharpen, IAAEmboss, RandomBrightnessContrast, OneOf, Compose
)

def aug_options(p=1):
    return Compose([
        Resize(224, 224),
        Compose([
            # RandomRotate90(),
            # Rotate(limit=270, interpolation=1, p=1),
            ShiftScaleRotate(shift_limit=0.2, rotate_limit=360,p=1, border_mode=1),
        ], p=1),
    ], p=p)

def apply_aug(aug, image):
    image = aug(image=image)['image']
    return image

image_path = '/home/barcelona/Image_augmentation/test_img/img1.jpg'
output_path = '/home/barcelona/Image_augmentation/output/alb_aug'

start_time = time.time()

image = cv2.imread(image_path)
aug = aug_options(p=1)
for i in range(0, 100):
    aug_image = apply_aug(aug, image)
    cv2.imwrite(output_path + '/' + str(time.time()) + '.jpg', aug_image)

print((time.time() - start_time)/60)