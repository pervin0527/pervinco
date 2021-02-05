import cv2, string
import numpy as np
import pandas as pd
from random import randrange, choice, sample, randint, shuffle

def get_mnist_letters():
    csv = pd.read_csv('/data/backup/pervinco/datasets/dirty_mnist_2/mnist_data_2nd/train.csv')
    images = csv.drop(['id', 'digit', 'letter'], axis=1).values
    images = images.reshape(-1, 28, 28, 1)
    # images = np.where((images <= 20) & (images != 0), 0., images)

    alphabets = list(string.ascii_uppercase)
    labels = list(csv['letter'])

    for idx, value in enumerate(labels):
        labels[idx] = alphabets.index(value)

    total_ds = []
    for image, label in zip(images, labels):
        image = image.astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        total_ds.append((image, label))

    return total_ds
    


def make_background():
    bg_size = (256, 256, 3)
    background = np.zeros(bg_size, np.uint8)

    return background


def make_coordinate(num):
    xp = [x for x in range(0, 256-16, 20)]
    yp = [y for y in range(0, 256-16, 20)]

    return xp, yp


def overlay(foreground, num_outputs):
    for i in range(num_outputs):
        num_letters = randrange(10, 12)

        fg = sample(foreground, num_letters)
        bg = make_background()
        bg_height, bg_width = bg.shape[0], bg.shape[1]

        x_coords, y_coords = make_coordinate(num_letters)
        shuffle(x_coords)
        shuffle(y_coords)

        for idx, (fg_image, fg_label) in enumerate(fg):
            x, y = x_coords[idx], y_coords[idx]
            # print(x, y)
            
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
            
            # cv2.imshow('result', bg)
            # cv2.waitKey(0)

        cv2.imwrite(f'/data/backup/pervinco/datasets/dirty_mnist_2/test/{i}.png', bg)



if __name__ == "__main__":
    foreground = get_mnist_letters()
    overlay(foreground, 1000)