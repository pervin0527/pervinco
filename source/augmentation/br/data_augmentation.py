import cv2
import random
import numpy as np
import albumentations as A

from tqdm import tqdm
from glob import glob
from utils import make_file_list, load_annot_data, make_save_dir, annot_write


def basic_augmentation(current_files):
    idx = random.randint(0, len(current_files)-1)
    image_file, annot_file = current_files[idx]
    image = cv2.imread(image_file)
    bboxes, labels = load_annot_data(annot_file, target_classes=classes)

    transformed = basic_transform(image=image, bboxes=bboxes, labels=labels)
    aug_image, aug_bboxes, aug_labels = transformed["image"], transformed["bboxes"], transformed["labels"]

    return aug_image, aug_bboxes, aug_labels


def adjust_coordinates(bboxes):
    ## remove too small bbox(object)
    results = []
    for box in bboxes:
        if box[2] - box[0] < 5 or box[3] - box[1] < 5:
            continue
        results.append(box)
    results = np.array(results)

    return results


def get_background_image(dirs):
    mixup_file_list = []
    for dir in dirs:
        mixup_files = glob(f"{dir}/*.jpg")
        mixup_file_list.extend(mixup_files)
        
    idx = random.randint(0, len(mixup_file_list)-1)
    background_image = cv2.imread(mixup_file_list[idx])

    return background_image


def mixup_augmentation(fg_image, min=0.4, max=0.5, alpha=1.0):
    background_transform = A.Compose([
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=(-0.3, 0.3), p=0.8),
            A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=(0, 0), val_shift_limit=(0, 100), p=0.8),
        ], p=0.7),
        
        A.OneOf([
            A.RandomRotate90(p=0.4),
            A.HorizontalFlip(p=0.3),
            A.VerticalFlip(p=0.3),
        ], p=0.3),

        A.ChannelShuffle(p=0.3),
        # A.MotionBlur(blur_limit=(3, 7), p=0.3),
        A.RGBShift(p=0.3),

        A.Resize(height=fg_image.shape[0], width=fg_image.shape[1], p=1),
    ])

    lam = np.clip(np.random.beta(alpha, alpha), min, max)
    bg_image = get_background_image(mixup_data_dir)
    bg_transformed = background_transform(image=bg_image)
    bg_image = bg_transformed["image"]

    mixup_image = (lam * bg_image + (1 - lam) * fg_image).astype(np.uint8)

    return mixup_image



def crop_image(image, bboxes, labels, coordinates):
    crop_transform = A.Compose([
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=(-0.3, 0.3), p=0.8),
            A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=(0, 0), val_shift_limit=(0, 100), p=0.8),
        ], p=0.7),
        
        A.OneOf([
            A.RandomRotate90(p=0.4),
            A.HorizontalFlip(p=0.3),
            A.VerticalFlip(p=0.3),
        ], p=0.3),

        A.ChannelShuffle(p=0.3),
        # A.MotionBlur(blur_limit=(3, 7), p=0.3),
        A.RGBShift(p=0.3),

        A.OneOf([
            A.RandomSizedBBoxSafeCrop(img_size, img_size, p=0.4),
            A.CropAndPad(percent=0.2, pad_mode=0, keep_size=True, p=0.4),
            # A.RandomCrop(img_size, img_size, p=0.2),
        ],p=0.6),

        A.Resize(height=coordinates[3]-coordinates[1], width=coordinates[2]-coordinates[0], p=1),

    ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]))
    cropped = crop_transform(image=image, bboxes=bboxes, labels=labels)
    crop_image, crop_bboxes, crop_labels = cropped["image"], cropped["bboxes"], cropped["labels"]

    # if mixup and random.random() < mixup_prob:
    #     crop_image = mixup_augmentation(crop_image, min=mixup_min, max=mixup_max)  

    return crop_image, np.array(crop_bboxes), crop_labels


def mosaic_augmentation(current_files):
    mosaic_image = np.full((img_size, img_size, 3), 128, dtype=np.uint8)
    mosaic_boxes, mosaic_labels = [], []

    xc = int(random.uniform(img_size * 0.25, img_size * 0.75))
    yc = int(random.uniform(img_size * 0.25, img_size * 0.75))
    indices = [random.randint(0, len(current_files)-1) for _ in range(4)]
    for i, index in enumerate(indices):
        image_file, annotation_file = current_files[index]
        image = cv2.imread(image_file)
        bboxes, labels = load_annot_data(annotation_file, target_classes=classes)
        # bboxes = adjust_coordinates(bboxes)
        bboxes = np.array(bboxes)

        if i == 0:
            image, bbox, label = crop_image(image, bboxes, labels, (img_size-xc, img_size-yc, img_size, img_size))
            mosaic_image[0 : yc, 0 : xc, :] = image
            mosaic_boxes.extend(bbox)
            mosaic_labels.extend(label)

        elif i == 1:
            image, bbox, label = crop_image(image, bboxes, labels, (0, img_size-yc, img_size-xc, img_size))
            mosaic_image[0 : yc, xc : img_size, :] = image

            if bbox.shape[0] > 0:
                bbox[:, [0, 2]] += xc

            mosaic_boxes.extend(bbox)
            mosaic_labels.extend(label)

        elif i == 2:
            image, bbox, label = crop_image(image, bboxes, labels, (0, 0, img_size-xc, img_size-yc))
            mosaic_image[yc:img_size, xc:img_size, :] = image

            if bbox.shape[0] > 0:
                bbox[:, [0, 2]] += xc
                bbox[:, [1, 3]] += yc

            mosaic_boxes.extend(bbox)
            mosaic_labels.extend(label)
        
        else:
            image, bbox, label = crop_image(image, bboxes, labels, (img_size-xc, 0, img_size, img_size-yc))
            mosaic_image[yc : img_size, 0 : xc, :] = image

            if bbox.shape[0] > 0:
                bbox[ :, [1, 3]] += yc

            mosaic_boxes.extend(bbox)
            mosaic_labels.extend(label)

    return mosaic_image, mosaic_boxes, mosaic_labels

        
def train_augmentation(files):
    save_path = f"{save_dir}/train"
    make_save_dir(save_path)

    for number in tqdm(range(total_steps)):
        if mosaic and random.random() < mosaic_prob:
            result_image, result_bboxes, result_labels = mosaic_augmentation(files)
        else:
            result_image, result_bboxes, result_labels = basic_augmentation(files)

        if mixup and random.random() < mixup_prob:
            result_image = mixup_augmentation(result_image, mixup_min, mixup_max)

        cv2.imwrite(f"{save_path}/JPEGImages/{number:>09}.jpg", result_image)
        annot_write(f"{save_path}/Annotations/{number:>09}.xml", result_bboxes, result_labels, result_image.shape[:2])

        sample = result_image.copy()
        for bbox in result_bboxes:
            xmin, ymin, xmax, ymax = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            cv2.rectangle(sample, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
        cv2.imwrite(f"{save_path}/Results/{number:>09}.jpg", sample)

        # cv2.imshow("sample", sample)
        # cv2.waitKey(0)

def valid_augmentation(files):
    save_path = f"{save_dir}/valid"
    make_save_dir(save_path)

    for idx in tqdm(range(len(files))):
        random.shuffle(files)
        image_file, annot_file = files[idx]
        image = cv2.imread(image_file)
        bboxes, labels = load_annot_data(annot_file, target_classes=classes)

        val_transformed = valid_transform(image=image, bboxes=bboxes, labels=labels)
        val_image, val_bboxes, val_labels = val_transformed["image"], val_transformed["bboxes"], val_transformed["labels"]

        cv2.imwrite(f"{save_path}/JPEGImages/{idx:>09}.jpg", val_image)
        annot_write(f"{save_path}/Annotations/{idx:>09}.xml", val_bboxes, val_labels, val_image.shape[:2])

        sample = val_image.copy()
        for bbox in val_bboxes:
            xmin, ymin, xmax, ymax = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            cv2.rectangle(sample, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
        cv2.imwrite(f"{save_path}/Results/{idx:>09}.jpg", sample)

if __name__ == "__main__":
    data_dir = ["/home/ubuntu/Datasets/BR/seed1_384"]
<<<<<<< HEAD
    save_dir = "/home/ubuntu/Datasets/BR/set1_384"
=======
    save_dir = "/home/ubuntu/Datasets/BR/set1_384_test"
>>>>>>> 3b45de715aa72bce876c0f1c2757295d77d0b141
    total_steps = 300000
    num_valid = 1000
    classes = ["Baskin_robbins"] # Baskin_robbins

    img_size = 384
    mosaic = True
    mosaic_prob = 0.4
    mixup = True
    mixup_prob = 0.4
    mixup_min = 0.1
    mixup_max = 0.3
    mixup_data_dir = ["/home/ubuntu/Datasets/VOCdevkit/VOC2012/JPEGImages", "/home/ubuntu/Datasets/SPC/Background"]

    basic_transform = A.Compose([
        A.Resize(img_size, img_size, p=1),
        
        A.OneOf([
            A.RandomSizedBBoxSafeCrop(img_size, img_size, p=0.4),
            A.CropAndPad(percent=0.2, pad_mode=0, keep_size=True, p=0.4),
            # A.RandomCrop(img_size, img_size, p=0.2),
        ],p=0.4),

        A.OneOf([
            
        ], p=1),

        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=(-0.3, 0.3), p=0.8),
            A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=(0, 0), val_shift_limit=(0, 100), p=0.8),
        ], p=0.5),
        
        A.OneOf([
            A.HorizontalFlip(p=0.3),
            A.VerticalFlip(p=0.3),
        ], p=0.3),

        A.ChannelShuffle(p=0.3),
        # A.MotionBlur(blur_limit=(3, 7), p=0.3),
        A.RGBShift(p=0.3),

    ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]))

    valid_transform = A.Compose([
        A.Resize(img_size, img_size, p=1),
    ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]))

    train_files, valid_files = make_file_list(data_dir, num_valid)
    train_augmentation(train_files[:1000])
    valid_augmentation(valid_files)