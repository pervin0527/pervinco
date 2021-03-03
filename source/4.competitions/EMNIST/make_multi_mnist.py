import cv2, string, os
import numpy as np
import pandas as pd
import albumentations as A
from random import randrange, choice, sample, randint, shuffle
from sklearn.preprocessing import OneHotEncoder
from albumentations.core.transforms_interface import ImageOnlyTransform


def get_random_kernel():
    structure = np.random.choice([cv2.MORPH_RECT, cv2.MORPH_ELLIPSE, cv2.MORPH_CROSS])
    kernel = cv2.getStructuringElement(structure, tuple(np.random.randint(1, 6, 2)))
    return kernel
    

def opening(img):
    img = cv2.erode(img, get_random_kernel(), iterations=1)
    img = cv2.dilate(img, get_random_kernel(), iterations=1)
    return img


def closing(img):
    img = cv2.dilate(img, get_random_kernel(), iterations=1)
    img = cv2.erode(img, get_random_kernel(), iterations=1)
    return img


class Closing(ImageOnlyTransform):
    def __init__(self, always_apply=False, p=1):
        super(Closing, self).__init__(always_apply, p)

    def apply(self, img, **params):
        return closing(img)


class Opening(ImageOnlyTransform):
    def __init__(self, always_apply=False, p=1):
        super(Opening, self).__init__(always_apply, p)
        
    def apply(self, img, **params):
        return opening(img)


def get_mnist_letters():
    csv = pd.read_csv(input_path)
    images = csv.drop(['id', 'digit', 'letter'], axis=1).values
    images = images.reshape(-1, 28, 28, 1)
    images = np.where((images > 20) & (images != 0), 255, images)
    images = np.where((images <= 20) & (images != 0), 0, images)

    CLASSES = list(string.ascii_uppercase)
    labels = list(csv['letter'])

    for idx, value in enumerate(labels):
        labels[idx] = CLASSES.index(value)

    total_ds = []
    for image, label in zip(images, labels):
        image = image.astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        total_ds.append((image, label))

    return total_ds, CLASSES
    


def make_background():
    bg_size = (256, 256, 3)
    background = np.zeros(bg_size, np.uint8)

    return background


def make_coordinate(num):
    xp = [x for x in range(0, 256-36, 20)]
    yp = [y for y in range(0, 256-36, 20)]

    return xp, yp


def overlay(foreground, num_outputs):
    label_df = []
    for i in range(num_outputs):
        num_letters = randrange(1, 12)

        check = []
        fg = []
        for _ in range(num_letters):
            ch = randint(0, len(foreground)-1)
            data = foreground[ch]

            if data[1] in check:
                pass
            else:
                check.append(data[1])
                fg.append(data)

        bg = make_background()
        bg_height, bg_width = bg.shape[0], bg.shape[1]

        x_coords, y_coords = make_coordinate(num_letters)
        shuffle(x_coords)
        shuffle(y_coords)

        labels = np.zeros([len(CLASSES)], dtype=np.float)

        for idx, (fg_image, fg_label) in enumerate(fg):
            x, y = x_coords[idx], y_coords[idx]
            # print(x, y)

            IMG_RESIZE = randrange(45, 55)
            transforms = A.Compose([
                A.Resize(IMG_RESIZE, IMG_RESIZE, p=1),
                A.HorizontalFlip(p=0.7),
                A.VerticalFlip(p=0.7),
                A.RandomRotate90(p=0.9),
                Closing(p=0.7),
                Opening(p=0.7)
            ])
            fg_image = transforms(image=fg_image)['image']
            
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

            bg[y : y + fg_height, x : x + fg_width] = (1.0 - mask) * bg[y : y + fg_height, x : x + fg_width] + mask * overlay_image
            labels[fg_label] += 1

        if not os.path.isdir(f'{output_path}/custom_multi'):
            os.makedirs(f'{output_path}/custom_multi')
        
        # cv2.imshow('result', result)
        # cv2.waitKey(0)

        # cv2.imwrite(f'{output_path}/custom_multi/{i}.png', bg)
        # label_df.append(labels)

        noise_factor = 0.75
        result = bg + noise_factor * np.random.normal(loc=0., scale=1.0, size=bg.shape)
        result = np.uint8(result)
        result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        # den = np.where((result <= 254) & (result != 0), 0, result)
        
        # cv2.imshow('result', result)
        # cv2.imshow('den', den)
        # cv2.waitKey(0)

        cv2.imwrite(f'{output_path}/custom_multi/{i}.png', result)
        label_df.append(labels)

    return label_df
        

if __name__ == "__main__":
    input_path = '/data/backup/pervinco/datasets/dirty_mnist_2/mnist_data_2nd/train.csv'
    output_path = f'/data/backup/pervinco/test_code/'
    foreground, CLASSES = get_mnist_letters()
    CLASSES = list(map(str.lower, CLASSES))
    result_df = overlay(foreground, 2000)

    result_df = pd.DataFrame(result_df)
    result_df.to_csv(f'{output_path}/custom_multi.csv', index_label='index', header=CLASSES)