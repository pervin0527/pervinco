import os
import cv2
import random
import albumentations as A
from glob import glob
from tqdm import tqdm


dataset_path = "/data/Datasets/Seeds/SPC/set10/"

images_path = f"{dataset_path}/images"
images = sorted(glob(f"{images_path}/*.jpg"))
random.shuffle(images)
print(len(images))


transform = A.Compose([
    A.Resize(448, 448, p=1),
    A.OneOf([
        A.Rotate(p=1, border_mode=0, limit=(-25, 25)),
        A.ShiftScaleRotate(p=1, border_mode=0, rotate_limit=(-25, 25)),
    ]),

    A.MotionBlur(p=0.55),

    A.OneOf([
        A.RandomContrast(p=0.7, limit=(-0.2, 0.2)),
        A.RandomBrightness(p=0.7, limit=(-0.1, 0.25))
    ], p=0.1),

    # A.Cutout(p=0.6, num_holes=24, max_w_size=16, max_h_size=16)
])


aug_per_img = 5
for i in tqdm(range(len(images))):
    image = images[i]
    filename = image.split('/')[-1].split('.')[0]
    image = cv2.imread(image)

    if not os.path.isdir(f"{dataset_path}/augmentations"):
        os.makedirs(f"{dataset_path}/augmentations")

    for idx in range(aug_per_img):
        augmented_image = transform(image=image)['image']
        cv2.imwrite(f"{dataset_path}/augmentations/{i}_{idx}.jpg", augmented_image)
        # cv2.imshow('result', augmented_image)
        # cv2.waitKey(0)