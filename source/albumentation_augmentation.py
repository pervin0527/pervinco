# -*- coding: utf-8 -*-
import numpy as np
import time
import cv2
import random
import glob
import shutil
import os
from matplotlib import pyplot as plt
from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90, RandomGamma, VerticalFlip,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue, 
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, RandomBrightnessContrast, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, Flip, OneOf, Compose, Rotate, RandomContrast, RandomBrightness, RandomCrop, Resize, OpticalDistortion
)


def aug_options(p=1):
    return Compose([
        Resize(224, 224),
RandomCrop(224,224, p=0.5),  # 위에꺼랑 세트
        
        OneOf([
        RandomContrast(p=1, limit=(-0.5,2)),   # -0.5 ~ 2 까지가 현장과 가장 비슷함  -- RandomBrightnessContrast
        RandomBrightness(p=1, limit=(-0.2,0.4)),
        RandomGamma(p=1, gamma_limit=(80,200)),
        ], p=0.6),
            
        OneOf([
            Rotate(limit=30, p=0.3),
            RandomRotate90(p=0.3),
            VerticalFlip(p=0.3)
        ], p=0.3),
    
        MotionBlur(p=0.2),   # 움직일때 흔들리는 것 같은 이미지
        ShiftScaleRotate(shift_limit=0.001, scale_limit=0.1, rotate_limit=30, p=0.3, border_mode=1),
        Resize(224,224, p=1),
        ],
        p=p)


def apply_aug(aug, image):
    image = aug(image=image)['image']
    return image


def img_distribution(path):
    labels = sorted(os.listdir(path))
    # print(labels)

    imgs = []
    for i in labels:
        # print(i)
        num_imgs = sorted(glob.glob(path + '/' + i + '/*.jpg'))
        imgs.append(len(num_imgs))

    # print(imgs)

    avg = int(sum(imgs, 0.0) / len(imgs))
    print(avg)

    plt.figure(figsize=(14,6))
    plt.barh(labels, imgs)
    plt.title('Num of images Distribution')
    plt.xlabel('NUM OF IMAGES')
    plt.ylabel('CLASS_NAMES')
    plt.show()


def show_aug_sampels(path):
    imgs = glob.glob(path + '/*/*.jpg')
    idx = random.randint(0, len(imgs))

    for i in range(0, 10):
        image = cv2.imread(imgs[idx])
        aug = aug_options(p=1)
        aug_img = apply_aug(aug, image)

        numpy_horizontal = np.hstack((image, aug_img))
        numpy_horizontal_concat = np.concatenate((image, aug_img), axis=1)
        numpy_horizontal_concat = cv2.resize(numpy_horizontal_concat, (1280, 720))

        cv2.imshow('Original / Augmentation', numpy_horizontal_concat)
        cv2.waitKey(300)

    



if __name__ == "__main__":
    path = '/data/backup/pervinco_2020/datasets/smart_shelf_beverage'
    dataset_name = path.split('/')[-1]
    output_path = '/data/backup/pervinco_2020/datasets/Auged_dataset/' + dataset_name

    img_distribution(path)
    dataset_path = sorted(glob.glob(path + '/*'))

    print("Num of Labels : ", len(dataset_path))

    show_aug_sampels(path)

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
                # cv2.waitKey(200)

                if not(os.path.isdir(output_path + '/' + label)):
                    os.makedirs(os.path.join(output_path + '/' + label))

                else:
                    pass
                # print(file_name, idx)
                cv2.imwrite(output_path + '/' + label + '/aug_' + str(idx) + '_' + file_name, aug_image)
                idx += 1
