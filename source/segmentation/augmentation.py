import os
import cv2
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
from random import randint
from sklearn.model_selection import train_test_split

def create_save_dir(path):
    if not os.path.isdir(path):
        os.makedirs(f"{path}/images")
        os.makedirs(f"{path}/masks")


def decode_segmentation_masks(mask, colormap, n_classes):
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    for l in range(0, n_classes):
        idx = mask == l
        r[idx] = colormap[l, 0]
        g[idx] = colormap[l, 1]
        b[idx] = colormap[l, 2]
    rgb = np.stack([r, g, b], axis=2)

    return rgb


def get_overlay(image, colored_mask):
    image = np.array(image).astype(np.uint8)
    overlay = cv2.addWeighted(image, 0.35, colored_mask, 0.65, 0)

    return overlay


def visualize(display_list):
    plt.figure(figsize=(15, 15))

    for i in range(len(display_list)):
        if display_list[i].shape[-1] == 3:
            display_list[i] = cv2.cvtColor(display_list[i], cv2.COLOR_BGR2RGB)
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


def rand_bbox(size, lamb):
    W = size[0]
    H = size[1]
    cut_rat = np.sqrt(1. - lamb)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)

    return x1, y1, x2, y2


def cutmix(image1, mask1, beta=1.0):
    idx = np.random.randint(0, len(images))
    image2, mask2 = images[idx], masks[idx]
    image2 = cv2.imread(image2)
    mask2 = cv2.imread(mask2, cv2.IMREAD_GRAYSCALE)

    cutmix_transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE, p=1),
        A.RandomRotate90(p=0.5)
    ])

    transform1 = cutmix_transform(image=image1, mask=mask1)
    transform2 = cutmix_transform(image=image2, mask=mask2)

    transform_image1, transform_mask1 = transform1['image'], transform1['mask']
    transform_image2, transform_mask2 = transform2['image'], transform2['mask']

    lam = np.random.beta(beta, beta)
    x1, y1, x2, y2 = rand_bbox(transform_image1.shape, lam)

    cutmix_image, cutmix_mask = transform_image1.copy(), transform_mask1.copy()
    cutmix_image[x1:x2, y1:y2, :] = transform_image2[x1:x2, y1:y2, :]
    cutmix_mask[x1:x2, y1:y2] = transform_mask2[x1:x2, y1:y2]

    # decode_mask = decode_segmentation_masks(cutmix_mask, colormap, len(colormap))
    # overlay = get_overlay(cutmix_image, decode_mask)
    # visualize([cutmix_image, cutmix_mask, overlay])

    return cutmix_image, cutmix_mask


def augmentation(images, masks, is_train):
    if is_train:
        save_path = f"{output_path}/train"
        create_save_dir(save_path)

        for i in tqdm(range(len(images))):
            image, mask = images[i], masks[i]
            file_name = image.split('/')[-1].split('.')[0]

            image = cv2.imread(image)
            mask = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
        
            for idx in range(ITER):
                number = randint(0, 1)
                # print(number)

                if number == 0:
                    transformed = train_transform(image=image, mask=mask)
                    result_image, result_mask = transformed['image'], transformed['mask']

                elif number == 1:
                    result_image, result_mask = cutmix(image, mask)

                if not VISUAL:
                    cv2.imwrite(f"{save_path}/images/{file_name}_{idx}.jpg", result_image)
                    cv2.imwrite(f"{save_path}/masks/{file_name}_{idx}.png", result_mask)

                else:
                    decode_mask = decode_segmentation_masks(result_mask, colormap, len(colormap))
                    overlay = get_overlay(result_image, decode_mask)
                    visualize([cv2.resize(image, (IMG_SIZE, IMG_SIZE)), cv2.resize(mask, (IMG_SIZE, IMG_SIZE)), result_image, result_mask, overlay])

    else:
        save_path = f"{output_path}/valid"
        create_save_dir(save_path)

        for i in tqdm(range(len(images))):
            image, mask = images[i], masks[i]
            file_name = image.split('/')[-1].split('.')[0]

            image = cv2.imread(image)
            mask = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)

            transformed = valid_transform(image=image, mask=mask)
            valid_image, valid_mask = transformed['image'], transformed['mask']

            cv2.imwrite(f"{save_path}/images/{file_name}.jpg", valid_image)
            cv2.imwrite(f"{save_path}/masks/{file_name}.png", valid_mask)


if __name__ == "__main__":
    root = "/data/Datasets/VOCtrainval_11-May-2012/VOCdevkit/VOC2012"
    image_path = f"{root}/JPEGImages"
    mask_path = f"{root}/SegmentationRaw"
    output_path = f"{root}/TEST"

    ITER = 2
    IMG_SIZE = 320
    VISUAL = False
    images, masks = voc_get_files(mask_path)
    print(len(images), len(masks))
    train_images, valid_images, train_masks, valid_masks = train_test_split(images, masks, test_size=0.1, shuffle=True)

    colormap = [[0, 0, 0],
                [128, 0, 0],
                [0, 128, 0],
                [128, 128, 0],
                [0, 0, 128],
                [128, 0, 128],
                [0, 128, 128],
                [128, 128, 128],
                [64, 0, 0],
                [192, 0, 0],
                [64, 128, 0],
                [192, 128, 0],
                [64, 0, 128],
                [192, 0, 128],
                [64, 128, 128],
                [192, 128, 128],
                [0, 64, 0],
                [128, 64, 0],
                [0, 192, 0],
                [128, 192, 0],
                [0, 64, 128]]
    colormap = np.array(colormap, dtype=np.uint8)

    train_transform = A.Compose([
            # A.PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE, border_model=0, p=0.1),
            A.Resize(IMG_SIZE, IMG_SIZE, p=1, always_apply=True),
            A.RandomSizedCrop(min_max_height=(IMG_SIZE/2, IMG_SIZE), height=IMG_SIZE, width=IMG_SIZE, p=0.5),

            A.OneOf([
                A.VerticalFlip(p=0.3),
                A.HorizontalFlip(p=0.3),
                A.Transpose(p=0.3)
            ], p=0.5),

            A.OneOf([
                A.ShiftScaleRotate(p=0.25, border_mode=0),
                A.RandomRotate90(p=0.25),
                A.OpticalDistortion(p=0.25, distort_limit=0.85, shift_limit=0.85, mask_value=0, border_mode=0),
                A.GridDistortion(p=0.25, distort_limit=0.85, mask_value=0, border_mode=0)
            ], p=1),
            
            A.OneOf([
                A.Blur(blur_limit=3, p=0.5),
                A.MotionBlur(blur_limit=3, p=0.5)
            ], p=0.4),

            A.OneOf([
                A.RandomBrightnessContrast(p=0.5),
                A.HueSaturationValue(p=0.5),
            ], p=1),

            A.OneOf([
                A.GridDropout(fill_value=0, mask_fill_value=0, random_offset=True, p=0.5),
                A.CoarseDropout(min_holes=32, max_holes=64,
                                min_height=16, min_width=16,
                                max_height=28, max_width=28,
                                fill_value=0, mask_fill_value=0,
                                p=0.5),
            ], p=0.6),

    ])

    valid_transform = A.Compose([
            A.Resize(IMG_SIZE, IMG_SIZE, p=1)
    ])

    augmentation(train_images, train_masks, True)
    augmentation(valid_images, valid_masks, False)