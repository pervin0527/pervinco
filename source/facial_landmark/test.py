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
        cv2.circle(image, (int(x), int(y)), radius=1, color=(255, 0, 0), thickness=3)

    return image

def read_data(annotations, save_dir=None):
    if save_dir:
        make_save_dir(save_dir)

    for idx, tmp in enumerate(annotations):
        keypoints = tmp[0 : 196]
        bboxes = tmp[196 : 200]
        image = cv2.imread(f"{image_dir}/{tmp[206]}")

        keypoints = keypoints.reshape((98, 2))
        xy = np.min(keypoints, axis=0).astype(np.int32) 
        zz = np.max(keypoints, axis=0).astype(np.int32)
        wh = zz - xy + 1

        center = (xy + wh/2).astype(np.int32)
        boxsize = int(np.max(wh)*1.2)
        xy = center - boxsize//2
        x1, y1 = xy
        x2, y2 = xy + boxsize
        height, width, _ = image.shape
        dx = max(0, -x1)
        dy = max(0, -y1)
        x1 = max(0, x1)
        y1 = max(0, y1)

        edx = max(0, x2 - width)
        edy = max(0, y2 - height)
        x2 = min(width, x2)
        y2 = min(height, y2)

        imgT = image[y1:y2, x1:x2]
        if (dx > 0 or dy > 0 or edx > 0 or edy > 0):
            imgT = cv2.copyMakeBorder(imgT, dy, edy, dx, edx, cv2.BORDER_CONSTANT, 0)

        if imgT.shape[0] == 0 or imgT.shape[1] == 0:
            imgTT = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            for x, y in (keypoints+0.5).astype(np.int32):
                cv2.circle(imgTT, (x, y), 1, (0, 0, 255))
            cv2.imshow('0', imgTT)
            if cv2.waitKey(0) == 27:
                exit()

        imgT = cv2.resize(imgT, (IMG_SIZE, IMG_SIZE))
        keypoints = (keypoints - xy)/boxsize
        print(keypoints.shape)
        print(keypoints)

        for (x, y) in keypoints.astype(np.int32):
            print(x + x1, y + y1)

        # cv2.imwrite(f"{save_dir}/images/{idx:>05}.jpg", imgT)
        # with open(f"{save_dir}/keypoints/{idx:>05}.txt", "w") as f:
        #     for idx, (x, y) in enumerate(keypoints):
        #         if idx == 97:
        #             f.write(f"{x},{y}")
        #         else:
        #             f.write(f"{x},{y},")
        # f.close()
        break



if __name__ == "__main__":
    IMG_SIZE = 112
    VISUALIZE = False
    image_dir = "/data/Datasets/WFLW/WFLW_images"
    
    train_annotation_dir = "/data/Datasets/WFLW/WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_train.txt"
    test_annotation_dir = "/data/Datasets/WFLW/WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_test.txt"

    train_annotation = read_annotations(train_annotation_dir)
    read_data(train_annotation, save_dir="/data/Datasets/WFLW/train")

    test_annotation = read_annotations(test_annotation_dir)
    read_data(test_annotation, save_dir="/data/Datasets/WFLW/test")