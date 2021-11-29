import cv2
import albumentations as A
from glob import glob


dataset_path = "/data/Datasets/Seeds/SPC/set8/images"
images = sorted(glob(f"{dataset_path}/*.jpg"))
print(len(images))


transform = A.Compose([
    A.OneOf([
        A.Rotate(p=1),
        A.ShiftScaleRotate(p=1)
    ]),

    A.MotionBlur(p=0.7),
    # A.ChannelShuffle(p=0.6),

    # A.OneOf([
    #     A.HorizontalFlip(p=1),
    #     A.VerticalFlip(p=1)
    # ]),

    A.OneOf([
        A.RandomContrast(p=0.7, limit=(-0.5, 0.3)),
        A.RandomBrightness(p=0.7, limit=(-0.2, 0.3))
    ], p=0.1),

    # A.Cutout(p=0.)
])


aug_per_img = 3
for image in images:
    filename = image.split('/')[-1].split('.')[0]
    image = cv2.imread(image)

    for idx in range(aug_per_img):
        augmented_image = transform(image=image)['image']
        cv2.imwrite(f"/data/Datasets/Seeds/SPC/set9/img_aug/{filename}_{idx}.jpg", augmented_image)