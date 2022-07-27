import os
import cv2
import numpy as np
import albumentations as A

from tqdm import tqdm
from glob import glob


def mixup(fg, min=0.4, max=0.5, alpha=1.0):
    fg_height, fg_width = fg.shape[:2]
    lam = np.clip(np.random.beta(alpha, alpha), min, max)

    bg_transform = A.Compose([
        A.Resize(width=fg_width, height=fg_height, always_apply=True),

        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=(-0.4, 0.4), p=0.3),
            A.HueSaturationValue(val_shift_limit=(40, 80), p=0.3),
            A.ChannelShuffle(p=0.3)
        ], p=1),
    ])

    bg_files = glob(f"{MIXUP_DIR}/*")
    random_idx = np.random.randint(0, len(bg_files))
    bg_image = cv2.imread(bg_files[random_idx])
    bg_result = bg_transform(image=bg_image)
    bg_image = bg_result['image']

    result_image = (lam * bg_image + (1 - lam) * fg).astype(np.uint8)

    return result_image


def read_txt(txt):
    lines = open(txt, "r").readlines()
    
    new_lines = []
    for index in tqdm(range(len(lines))):
        line = lines[index]
        line = line.strip().split()

        img_path = line[0]
        image = cv2.imread(img_path)
        landmarks = np.array(line[1:137], dtype=np.float32)
        attributes = np.array(line[137:143], dtype=np.int32)
        euler_angles = np.array(line[143:146], dtype=np.float32)

        name = f"normal_{index:>07}.png"
        landmark_str = ' '.join(list(map(str, landmarks.reshape(-1).tolist())))
        attributes_str = ' '.join(list(map(str, attributes)))
        euler_angles_str = ' '.join(list(map(str, euler_angles)))
        cv2.imwrite(f"{SAVE_DIR}/imgs/{name}", image)
        label = f"{SAVE_DIR}/imgs/{name} {landmark_str} {attributes_str} {euler_angles_str}\n"
        new_lines.append(label)

        opt = np.random.randint(0, 2)

        if opt == 0:
            transformed = TRANSFORM(image=image)
            augment_img = transformed["image"]

        elif opt == 1:
            augment_img = mixup(image, min=0.2, max=0.25)

        keypoints = landmarks.reshape(-1, 2) * [IMG_SIZE, IMG_SIZE]

        if VISUALIZE:
            sample_img = augment_img.copy()
            for (x, y) in keypoints.astype(np.int32):
                cv2.circle(sample_img, (x, y), color=(0, 0, 255), radius=1, thickness=-1)

            cv2.imshow("result", sample_img)
            cv2.waitKey(0)

        name = f"augment_{index:>07}.png"
        landmark_str = ' '.join(list(map(str, landmarks.reshape(-1).tolist())))
        attributes_str = ' '.join(list(map(str, attributes)))
        euler_angles_str = ' '.join(list(map(str, euler_angles)))
        cv2.imwrite(f"{SAVE_DIR}/imgs/{name}", augment_img)
        label = f"{SAVE_DIR}/imgs/{name} {landmark_str} {attributes_str} {euler_angles_str}\n"
        new_lines.append(label)

    records = open(f"{SAVE_DIR}/list.txt", "w")
    for line in new_lines:
        records.writelines(line)


if __name__ == "__main__":
    TRAIN_TXT = "/data/Datasets/WFLW/train_data_68pts/list.txt"

    VISUALIZE = False
    IMG_SIZE = 112
    SAVE_DIR = "/data/Datasets/WFLW/augment_data_68pts"
    MIXUP_DIR = "/data/Datasets/Mixup_background"

    if not os.path.isdir(f"{SAVE_DIR}"):
        os.makedirs(f"{SAVE_DIR}/imgs")

    TRANSFORM = A.Compose([
        A.ChannelShuffle(p=0.3),

        A.OneOf([
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.5),
        ], p=1),

        A.OneOf([
            A.Blur(blur_limit=(3, 5), p=0.3),
            A.GaussNoise(var_limit=(10.0, 20.0), p=0.3),
            A.RandomRain(blur_value=2, p=0.3)
        ], p=0.4),

        A.CoarseDropout(max_holes=1, max_height=(IMG_SIZE // 2), max_width=(IMG_SIZE // 2), min_height=(IMG_SIZE // 4), min_width=(IMG_SIZE // 4), p=0.5)

    ])
    
    read_txt(TRAIN_TXT)