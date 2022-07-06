import os
import cv2
import numpy as np
import albumentations as A

from tqdm import tqdm
from glob import glob
from matplotlib import pyplot as plt


def flatten_landmark(landmark):
    flatten = []
    for point in landmark:
        x, y = point
        flatten.extend([x, y])

    return flatten


def draw_landmark(image, landmark, name):
    result = image.copy()
    for (x, y) in landmark:
        cv2.circle(result, (int(x), int(y)), radius=3, color=(0, 0, 255), thickness=-1)    

    # plt.figure(figsize=(8, 8))
    # plt.axis('off')
    # plt.imshow(result)
    # plt.show()    
    result = cv2.resize(result, (960, 720))
    cv2.imshow(f"{name}", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def augmentation(image, keypoints):
    result = TRANSFORM(image=image, keypoints=keypoints)
    result_image, result_keypoints = result['image'], result['keypoints']
    result_keypoints = np.array(flatten_landmark(result_keypoints), dtype=np.float32).reshape(-1, 2)

    return result_image, result_keypoints


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


def write_txt(labels):
    f = open(f"{SAVE_DIR}/annotations/list_68pt_rect_attr_train_test/list_68pt_rect_attr_train.txt", "w")
    for label in labels:
        f.writelines(label)


def read_text(text_file):
    if not os.path.isdir(f"{SAVE_DIR}"):
        os.makedirs(f"{SAVE_DIR}/WFLW_images")
        os.makedirs(f"{SAVE_DIR}/annotations/list_68pt_rect_attr_train_test")

    f = open(text_file, "r")
    lines = f.readlines()
    
    labels = []
    for index in tqdm(range(len(lines))):
        line = lines[index]
        line = line.strip().split()

        assert(len(line) == 147)
        landmark = np.array(line[:136], dtype=np.float32).reshape(-1, 2)
        bbox = np.array(line[136:140], dtype=np.int32)
        attributes = np.array(line[140:146], dtype=np.int32)
        image_path = line[146]

        image = cv2.imread(f"{IMG_DIR}/{image_path}")

        for step in range(STEPS):
            augment_image, augment_landmark = augmentation(image, landmark)

            if np.random.rand(1) > 0.5:
                augment_image = mixup(augment_image, min=0.2, max=0.4)
            
            file_name = f"{index}_{step:>06}.jpg"
            bbox_str = ' '.join(list(map(str, bbox)))
            landmark_str = ' '.join(list(map(str, augment_landmark.reshape(-1).tolist())))
            attributes_str = ' '.join(list(map(str, attributes)))

            if not VISUALIZE:
                label = f"{landmark_str} {bbox_str} {attributes_str} {SAVE_DIR}/WFLW_images/{file_name}\n"
                labels.append(label)
                cv2.imwrite(f"{SAVE_DIR}/WFLW_images/{file_name}", augment_image)

            else:
                draw_landmark(augment_image, augment_landmark, image_path)

    return labels


if __name__ == '__main__':
    ROOT_DIR = "/data/Datasets/WFLW"
    SAVE_DIR = "/data/Datasets/WFLW/custom"
    MIXUP_DIR = "/data/Datasets/Mixup_background"
    VISUALIZE = False
    STEPS = 1

    # black_list = ["50--Celebration_Or_Party/50_Celebration_Or_Party_houseparty_50_321.jpg"]
    black_list = []

    TRANSFORM = A.Compose([
        # A.Rotate(limit=(-20, 20), border_mode=0, p=0.5),
        # A.Resize(height=720, width=960, always_apply=True),

        # A.OneOf([
        #     A.Rotate(limit=(-20, 20), border_mode=0, p=0.5),
        #     A.ShiftScaleRotate(shift_limit=(-0.03, 0.03), scale_limit=(-0.0, 0.0), rotate_limit=(-20, 20), border_mode=0, p=0.5)
        # ], p=1),

        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=(-0.15, 0.15), contrast_limit=(-0.15, 0.15), p=0.5),
            A.HueSaturationValue(hue_shift_limit=(-0.15, 0.15), sat_shift_limit=(-0.15, 0.15), val_shift_limit=(.10, .10), p=0.5)
        ], p=1),

        # A.OneOf([
        #     A.VerticalFlip(p=0.5),
        #     A.HorizontalFlip(p=0.5),
        # ], p=0.3),

        A.OneOf([
            A.Blur(blur_limit=(3, 5), p=0.3),
            A.GaussNoise(var_limit=(10.0, 20.0), p=0.3),
            A.RandomRain(blur_value=2, p=0.3)
        ], p=0.4),

        A.RandomRotate90(p=0.3)

    ], keypoint_params=A.KeypointParams(format="xy", remove_invisible=False))
    
    IMG_DIR = f'{ROOT_DIR}/WFLW_images'
    TXT_DIR = f'{ROOT_DIR}/annotations/list_68pt_rect_attr_train_test/list_68pt_rect_attr_train.txt'
    labels = read_text(TXT_DIR)
    write_txt(labels)