# Explain ratio of p
# https://github.com/albumentations-team/albumentations/issues/586#issue-596422426
# Tutorial
# https://hoya012.github.io/blog/albumentation_tutorial/
# https://github.com/albumentations-team/albumentations_examples/blob/master/notebooks/example.ipynb
# API
# https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.RandomResizedCrop
# https://github.com/albumentations-team/albumentations
import os
import cv2
import time
import glob
from albumentations import (
    RandomCrop, HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90, Resize, Rotate,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, RandomSizedCrop, ChannelShuffle
)
aug_output_path = '/data/backup/pervinco_2020/datasets/cu50/alb_test/resized/'
crop_output_path = '/data/backup/pervinco_2020/datasets/cu50/alb_test/cropped/'

def aug_options(p=1):
    return Compose([
        Resize(224, 224),
        Compose([
            # RandomRotate90(),
            # Rotate(limit=270, interpolation=1, p=1),
            ShiftScaleRotate(rotate_limit=300,p=1),
        ], p=.8),
    ], p=p)


def aug_apply(aug, image):
    image = aug(image=image)['image']

    return image


ds_path = sorted(glob.glob('/data/backup/pervinco_2020/datasets/cu50/original_set/valid/*'))
total_imgs = 0
start_time = time.time()
for labels in ds_path:
    imgs = sorted(glob.glob(labels + '/*.jpg'))
    total_imgs += len(imgs)

    for img in imgs:
        class_name = img.split('/')[-2]

        if not (os.path.isdir(aug_output_path + '/' + class_name)):
            os.makedirs(aug_output_path + '/' + class_name)

        else:
            pass

        image = cv2.imread(img)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # cv2.imshow('test', image)
        # cv2.waitKey(0)
        aug = aug_options(p=1)
        for i in range(0, 50):
            aug_image = aug_apply(aug, image)
            cv2.imwrite(aug_output_path + class_name + '/' + str(time.time()) + '.jpg', aug_image)
            crop_image = aug_apply(RandomCrop(112, 112), aug_image)
            
            if not (os.path.isdir(crop_output_path + '/' + class_name)):
                os.makedirs(crop_output_path + '/' + class_name)

            else:
                pass

            cv2.imwrite(crop_output_path + class_name + '/' + str(time.time()) + '.jpg', crop_image)
            # cv2.imshow('original', image)
            # cv2.imshow('auged', aug_image)
            # cv2.imshow('cropped', crop_image)
            # cv2.waitKey(0)
    #         break
    # break

print(total_imgs)
print(time.time() - start_time)

