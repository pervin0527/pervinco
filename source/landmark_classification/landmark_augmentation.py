import sys
import glob
import cv2
import os
import math
import time
import pathlib
import datetime
import albumentations as A
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

def visualize(path, label_list):
    images = []
    for label in label_list:
        files = os.listdir(path + '/' + label)
        file_num = len(files)

        images.append(file_num)

    x, y = label_list, images
    plt.figure(figsize=(20, 10))
    plt.bar(x, y, width=0.9,)
    plt.xticks(x, rotation=90)
    plt.show()
        

def aug_visualize(labels, train_data_path, valid_data_path):
    train_images = []
    valid_images = []

    for label in labels:
        train_image_num = len(os.listdir(train_data_path + '/' + label))
        valid_image_num = len(os.listdir(valid_data_path + '/' + label))

        train_images.append(train_image_num)
        valid_images.append(valid_image_num)

    plt.figure(figsize=(20, 10))
    plt.bar(labels, train_images, width=0.9,)
    plt.bar(labels, valid_images, width=0.9,)
    plt.xticks(labels, rotation=90)
    plt.show() 


def basic_processing(ds_path):
    ds_path = pathlib.Path(ds_path)

    images = list(ds_path.glob('*/*'))
    images = [str(path) for path in images]

    train_images, valid_images = train_test_split(images, test_size=0.2)

    print(len(train_images), len(valid_images))

    return sorted(train_images), sorted(valid_images)


def split_seed(images, is_train, label_list, output_path, TODAY):
    if is_train:
        output_path = output_path + '/' + TODAY + '/train'

    else:
        output_path = output_path + '/' + TODAY + '/valid'

    for label in label_list:
        if not(os.path.isdir(output_path + '/' + label)):
            os.makedirs(output_path + '/' + label)

        else:
            pass

    for img in images:
        file_name = img.split('/')[-1]
        label = img.split('/')[-2]
        print(label, file_name, output_path)

        image = cv2.imread(img)
        transform = A.Resize(270, 480)
        augmented_image = transform(image = image)['image']
        cv2.imwrite(output_path + '/' + label + '/' + file_name, augmented_image)

    return output_path


def augmentation(set_path, label_list, aug_num):
    ds_path = pathlib.Path(set_path)

    ds = list(ds_path.glob('*'))
    ds = sorted([str(path) for path in ds])
    print(len(ds))
    
    for output_path in ds:
        label = output_path.split('/')[-1]

        images = os.listdir(output_path)
        n_images = len(images)
        cnt = int(math.ceil(aug_num / n_images))

        # print(label, n_images)
        
        for img in images:
            total = len(os.listdir(output_path))
            if total <= aug_num:
                file_name = img.split('/')[-1]
                # file_name = file_name.split('.')[0]
                image = cv2.imread(output_path + '/' + img)

                print(file_name)

                for c in range(cnt):
                    transform = A.Compose([
                                A.Resize(270, 480),
                                A.HorizontalFlip(p=0.5),
                                ])
                    augmented_image = transform(image=image)['image']
                    cv2.imwrite(output_path + '/' + 'aug_' + str(c) + file_name, augmented_image)
                
            else:
                pass

if __name__ == "__main__":
    TODAY = datetime.datetime.now().strftime("%Y.%m.%d_%H:%M:%S")
    print(TODAY)

    seed_path = sys.argv[1]
    aug_num = int(sys.argv[2])
    dataset_name = seed_path.split('/')[-1]
    output_path = '/data/backup/pervinco_2020/Auged_datasets/' + dataset_name

    label_list = sorted(os.listdir(seed_path + '/'))
    n_classes = len(label_list)

    visualize(seed_path, label_list)
    
    train_images, valid_images = basic_processing(seed_path)

    train_path = split_seed(train_images, True, label_list, output_path, TODAY)
    valid_path = split_seed(valid_images, False, label_list, output_path, TODAY)

    augmentation(train_path, label_list, aug_num)
    augmentation(valid_path, label_list, (aug_num * 0.2))

    aug_visualize(label_list, train_path, valid_path)
    