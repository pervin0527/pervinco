import cv2, pathlib, os, time
import albumentations as A
import numpy as np
import datetime
from random import randrange, choice, sample
from tqdm import tqdm

def get_foreground(fg_path):
    foreground = pathlib.Path(fg_path)
    classes = (list(foreground.glob('*')))
    classes = sorted([str(label).split('/')[-1] for label in classes])

    fg_list = []
    for label in classes:
        images = list(foreground.glob(label + '/*'))
        images = [str(image) for image in images]
        img_list = sample(images, 100)
        fg_list.append(img_list)

    return fg_list


def get_background(bg_path):
    background = pathlib.Path(bg_path)
    classes = (list(background.glob('*')))
    classes = sorted([str(label).split('/')[-1] for label in classes])

    bg_list = []
    for label in classes:
        images = list(background.glob(label + '/*'))
        images = [str(image) for image in images]
        image = choice(images)
        bg_list.append(image)

    return bg_list


def overlay(fg_list):
    for foreground in fg_list:
        for i in tqdm(range(len(foreground))):
            idx = 0
            RAND_SIZE = randrange(100, 130)
            transform = A.Resize(IMG_SIZE-30, RAND_SIZE)
            # x = randrange(0, 120)
            # y = randrange(50, 70)
            x = randrange(0, 120)
            y = randrange(0, 50)

            fg_image = foreground[i]
            fg_label = fg_image.split('/')[-2]
            fg_image = cv2.imread(fg_image)
            fg_image = transform(image=fg_image)['image']

            bg_list = get_background(bg_path)
            for bg_image in bg_list:
                bg_image = cv2.imread(bg_image)
                bg_image = cv2.resize(bg_image, (IMG_SIZE, IMG_SIZE))
                bg_height, bg_width = bg_image.shape[0], bg_image.shape[1]

                if x >= bg_width or y >= bg_height:
                    print("over size")
                    pass

                fg_height, fg_width = fg_image.shape[0], fg_image.shape[1]

                if x + fg_width > bg_width:
                    fg_width = bg_width - x
                    fg_image = fg_image[ :, : fg_width]

                if y + fg_height > bg_height:
                    fg_height = bg_height - y
                    fg_image = fg_image[ : fg_height]

                if fg_image.shape[2] < 4:
                    fg_image = np.concatenate([fg_image, np.ones((fg_image.shape[0], fg_image.shape[1], 1), dtype = fg_image.dtype) * 255], axis = 2)

                overlay_image = fg_image[..., : 3]
                mask = fg_image[..., 3:] / 255.0

                bg_image[y : y + fg_height, x : x + fg_width] = (1.0 - mask) * bg_image[y : y + fg_height, x : x + fg_width] + mask * overlay_image
                
                # if not os.path.isdir(f"{OUTPUT_PATH}/{fg_label}"):
                    # os.makedirs(f"{OUTPUT_PATH}/{fg_label}")
                # cv2.imwrite(f"{OUTPUT_PATH}/{fg_label}/overlay_{idx}_{time.time()}.jpg", bg_image)

                cv2.imshow("test", bg_image)
                cv2.waitKey(0)

                idx += 1
            

if __name__ == "__main__":
    IMG_SIZE = 224
    fg_path = "/data/backup/pervinco_2020/datasets/test_bev"
    bg_path = "/data/backup/pervinco_2020/datasets/test_snack"
    OUTPUT_PATH = "/data/backup/pervinco_2020/test_code/test_output"

    fg_list = get_foreground(fg_path)
    overlay(fg_list)