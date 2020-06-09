# -*- coding: utf-8 -*-
import numpy as np
import time
import cv2
import random
import glob
import shutil
import os

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

if __name__ == "__main__":
    dataset_path = sorted(glob.glob('/data/backup/pervinco_2020/datasets/data/original_train/*'))
    output_path = '/data/backup/pervinco_2020/datasets/data/aug'

    print(len(dataset_path))

    for labels in dataset_path:
        imgs = sorted(glob.glob(labels + '/*.jpg'))

        for img in imgs:
            file_name = img.split('/')[-1]
            label = img.split('/')[-2]

            image = cv2.imread(img)
            aug = aug_options(p=1)

            idx = 0
            for i in range(0, 10):
                aug_image = apply_aug(aug, image)
                # cv2.imshow('test', aug_image)
                # cv2.imshow('original', image)

                if not(os.path.isdir(output_path + '/' + label)):
                    os.makedirs(os.path.join(output_path + '/' + label))

                else:
                    pass
                print(file_name, idx)
                cv2.imwrite(output_path + '/' + label + '/aug_' + str(idx) + '_' + file_name, aug_image)
                idx += 1
