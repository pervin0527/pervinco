import os
import cv2
import albumentations as A
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def visualize(display_list):
    plt.figure(figsize=(30, 30))

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.imshow(display_list[i])
        plt.axis('off')
    plt.show()

def voc_get_files(mask_path):
    images = []
    masks = sorted(glob(f"{mask_path}/*.png"))
    
    for mask in masks:
        file_name = mask.split('/')[-1].split('.')[0]
        image = f"{image_path}/{file_name}.jpg"
        images.append(image)

    return images, masks

def augmentation(images, masks, is_train):
    if is_train:
        save_path = f"{output_path}/train"
        ITER = 5
    else:
        save_path = f"{output_path}/valid"
        ITER = 1

    if not os.path.isdir(save_path):
        os.makedirs(f"{save_path}/images")
        os.makedirs(f"{save_path}/masks")

    for i in tqdm(range(len(images))):
        image, mask = images[i], masks[i]
        file_name = image.split('/')[-1].split('.')[0]

        image = cv2.imread(image)
        mask = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # original_height, original_width = image.shape[:-1]

        if is_train:
            transform = A.Compose([
                A.Resize(IMG_SIZE, IMG_SIZE, p=1),
                # A.PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE, border_model=0, p=0.1),       

                A.OneOf([
                    A.Blur(blur_limit=3, p=0.5),
                    A.MotionBlur(blur_limit=3, p=0.5)
                ]),

                A.OneOf([
                    A.VerticalFlip(p=0.3),
                    A.HorizontalFlip(p=0.3),
                    A.Transpose(p=0.3)
                ], p=0.5),

                A.OneOf([
                    A.RandomBrightnessContrast(p=0.5),
                    A.HueSaturationValue(p=0.5),
                ], p=1),

                A.OneOf([
                    A.ShiftScaleRotate(p=0.5, border_mode=0),
                    A.RandomRotate90(p=0.5)
                ], p=1)
            ])

        else:
            transform = A.Compose([
                A.Resize(IMG_SIZE, IMG_SIZE, p=1)
            ])
    
        for idx in range(ITER):
            transformed = transform(image=image, mask=mask)
            transformed_image, transformed_mask = transformed['image'], transformed['mask']
            # visualize([image, mask, transformed_image, transformed_mask])

            cv2.imwrite(f"{save_path}/images/{file_name}_{idx}.jpg", transformed_image)
            cv2.imwrite(f"{save_path}/masks/{file_name}_{idx}.png", transformed_mask)

if __name__ == "__main__":
    image_path = "/data/Datasets/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages"
    mask_path = "/data/Datasets/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/SegmentationRaw"
    output_path = "/data/Datasets/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/Segmentation_Aug"

    IMG_SIZE = 512
    images, masks = voc_get_files(mask_path)
    print(len(images), len(masks))
    train_images, valid_images, train_masks, valid_masks = train_test_split(images, masks, test_size=0.1, shuffle=True, random_state=42)

    augmentation(train_images, train_masks, True)
    augmentation(valid_images, valid_masks, False)