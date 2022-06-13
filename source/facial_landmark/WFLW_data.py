## 0-195: landmark 좌표점   196-199: bbox 좌표점
## 200: 자세(pose)          0->정상자세(normal pos)e)           1->큰 자세(large pose)
## 201: 표정(expression)    0->정상표정(normal exp)ression)     1->과장된 표정(exaggerate expression)
## 202: 조도(illumination)  0->정상조명(normal ill)umination)   1->극단조명(extreme illumination)
## 203: 메이크업(make-up)   0->노메이크업(no make-up)            1->메이크업(make-up)
## 204: 가림(occlusion)     0->가림(noocclusio) 없음n)          1->가림(occlusion)
## 205: 블러(blur)          0->또렷(clear)                      1->블럭(blur)
## 206: 이미지 이름

import os
import cv2
import random
import numpy as np
import pandas as pd
import albumentations as A


def make_save_dir(dir):
    if not os.path.isdir(dir):
        os.makedirs(f"{dir}/images")
        os.makedirs(f"{dir}/keypoints")


def read_annotations(csv_path):
    df = pd.read_csv(csv_path, sep=' ', header=None, index_col=False)
    df = df.to_numpy()

    return df


def visualize(image, keypoints):
    for (x, y) in keypoints:
        cv2.circle(image, (int(x), int(y)), radius=1, color=(255, 0, 0), thickness=2,)

    return image


def make_valid_data(img, box, points, idx, save_dir):
    height, width = img.shape[:-1]
    padding_values = [10, 15, 20, 25, 30]
    padd = random.choice(padding_values)
    
    xmin, ymin, xmax, ymax = box
    xmin = max(0, xmin-padd)
    ymin = max(0, ymin-padd)
    xmax = min(width, xmax+padd)
    ymax = min(height, ymax+padd)

    points[points < 0] = 0
    points = points.reshape((98, 2))
    for i in range(len(points)):
        x, y = points[i][0], points[i][1]
        points[i][0] = min(width-1, x)
        points[i][1] = min(height-1, y)

    valid_transform = A.Compose([
        A.Crop(x_min=xmin, y_min=ymin, x_max=xmax, y_max=ymax, always_apply=True),
        A.Resize(IMG_SIZE, IMG_SIZE, always_apply=True),
    ], keypoint_params=A.KeypointParams(format="xy", remove_invisible=False))

    transformed = valid_transform(image=img, keypoints=points)
    print(len(points), len(transformed['keypoints']))

    if not VISUALIZE:
        cv2.imwrite(f"{save_dir}/images/{idx:>05}.jpg", transformed['image'])
        with open(f"{save_dir}/keypoints/{idx:>05}.txt", "w") as f:
            for n, (x, y) in enumerate(transformed['keypoints']):
                if n == 97:
                    f.write(f"{x},{y}")
                else:
                    f.write(f"{x},{y},")
        f.close()

    else:
        transform_image = visualize(transformed['image'], keypoints=transformed['keypoints'])
        ground_truth = visualize(img, points.reshape((98, 2)))
        cv2.imshow("ground_truth", cv2.resize(ground_truth, (512, 512)))
        cv2.imshow("augmentation", transform_image)
        cv2.waitKey(0)


def make_train_data(img, box, points, idx, number, save_dir):
    height, width = img.shape[:-1]
    padding_values = [10, 15, 20, 25, 30]
    padd = random.choice(padding_values)
    
    xmin, ymin, xmax, ymax = box
    xmin = max(0, xmin-padd)
    ymin = max(0, ymin-padd)
    xmax = min(width, xmax+padd)
    ymax = min(height, ymax+padd)

    points[points < 0] = 0
    points = points.reshape((98, 2))
    for i in range(len(points)):
        x, y = points[i][0], points[i][1]
        points[i][0] = min(width-1, x)
        points[i][1] = min(height-1, y)

    train_transform = A.Compose([
        A.Crop(x_min=xmin, y_min=ymin, x_max=xmax, y_max=ymax, always_apply=True),
        A.Resize(IMG_SIZE, IMG_SIZE, always_apply=True),

        A.OneOf([
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.5)
        ], p=0.6),

        A.OneOf([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5)
        ], p=0.45),

        A.RandomRotate90(p=0.5)
    ], keypoint_params=A.KeypointParams(format="xy", remove_invisible=False))

    transformed = train_transform(image=img, keypoints=points)
    print(len(points), len(transformed['keypoints']))

    if not VISUALIZE:
        cv2.imwrite(f"{save_dir}/images/{idx:>05}_{number}.jpg", transformed['image'])
        with open(f"{save_dir}/keypoints/{idx:>05}_{number}.txt", "w") as f:
            for n, (x, y) in enumerate(transformed['keypoints']):
                if n == 97:
                    f.write(f"{x},{y}")
                else:
                    f.write(f"{x},{y},")
        f.close()

    else:
        transform_image = visualize(transformed['image'], keypoints=transformed['keypoints'])
        ground_truth = visualize(img, points.reshape((98, 2)))
        cv2.imshow("ground_truth", cv2.resize(ground_truth, (512, 512)))
        cv2.imshow("augmentation", transform_image)
        cv2.waitKey(0)


def read_data(annotations, save_dir=None, is_train=False):
    if save_dir:
        make_save_dir(save_dir)

    for idx, tmp in enumerate(annotations):
        keypoints = tmp[0 : 196]
        bboxes = tmp[196 : 200]
        image = cv2.imread(f"{image_dir}/{tmp[206]}")
        
        if is_train:
            for step in range(30):
                make_train_data(image, bboxes, keypoints, idx, step, save_dir)

        else:
            make_valid_data(image, bboxes, keypoints, idx, save_dir)


if __name__ == "__main__":
    IMG_SIZE = 224
    VISUALIZE = False
    image_dir = "/home/ubuntu/Datasets/WFLW/WFLW_images"
    
    train_annotation_dir = "/home/ubuntu/Datasets/WFLW/WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_train.txt"
    test_annotation_dir = "/home/ubuntu/Datasets/WFLW/WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_test.txt"

    train_annotation = read_annotations(train_annotation_dir)
    read_data(train_annotation, save_dir="/home/ubuntu/Datasets/WFLW/train", is_train=False)

    test_annotation = read_annotations(test_annotation_dir)
    read_data(test_annotation, save_dir="/home/ubuntu/Datasets/WFLW/test", is_train=False)