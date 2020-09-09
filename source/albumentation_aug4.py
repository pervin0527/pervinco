import cv2
import glob
import sys
import os
import math
import random
import time
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import albumentations as A
import pandas as pd

def visualize(image):
    cv2.imshow('visualize', image)
    cv2.waitKey(0)


def make_df(path):
    result = []
    idx = 0
    label_list = sorted(os.listdir(path))
    print(label_list)

    for label in label_list:
        file_list = glob.glob(os.path.join(path,label,'*'))
        
        for file in file_list:
            result.append([idx, label, file])
            idx += 1
            
    img_df = pd.DataFrame(result, columns=['idx','label','image_path'])

    return img_df


def show_img_distribution(img_df):
    print(img_df['label'].value_counts().sort_index())
    img_df['label'].value_counts().sort_index().plot.barh(figsize=(14,10), title='Num of images Distribution')
    plt.xlabel('NUM OF IMAGES')
    plt.ylabel('CLASS_NAMES')
    plt.show()

    os.system("clear")


def show_splited_datasets(train_set, valid_set):
    labels = train_set['label'].sort_index()
    train_set_imgs = train_set['label'].value_counts().sort_index()
    valid_set_imgs = valid_set['label'].value_counts().sort_index()

    print("Train Set Distribution \n", train_set_imgs)
    print('=======================================================================')
    print("Validation Set Distribution \n", valid_set_imgs)

    index = []
    for l in labels:
        if l in index:
            pass
        else:
            index.append(l)

    train_imgs_num = []
    valid_imgs_num = []

    for i in train_set_imgs:
        train_imgs_num.append(i)

    for i in valid_set_imgs:
        valid_imgs_num.append(i)

    df = pd.DataFrame({'train imgs':train_imgs_num, 'valid imgs':valid_imgs_num}, index=index)

    df.plot.barh(title='Num of images Distribution', figsize=(14, 10))
    plt.show()

    os.system("clear")


def aug_processing(data_set, output_path, aug_num, is_train):
    img_path = data_set['image_path'].sort_index()
    labels = data_set['label'].value_counts().sort_index()

    if is_train == True:
        output_path = output_path + '/train_test'

    else:
        output_path = output_path + '/valid_test'

    for path in img_path:
        file_name = path.split('/')[-1]
        print(file_name)
        file_name = file_name.split('.')[0]
        label = path.split('/')[-2]
        avg = int(math.ceil(aug_num / labels[label]))

        image = cv2.imread(path)

        if not(os.path.isdir(output_path + '/' + label)):
            os.makedirs(output_path + '/' + label)

        else:
            pass

        total_auged = len(glob.glob(output_path + '/' + label + '/*.jpg'))

        if total_auged <= aug_num:
            for i in range(avg):
                transform = A.Compose([
                    A.Resize(224, 224, p=1),
                    # A.Rotate(limit=(-360, 360), p=0.5, border_mode=1),

                    # A.OneOf([
                    #     A.Rotate(limit=(-360, 360), p=0.5, border_mode=1),
                    #     A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50, rotate_limit=45, p=.5),
                    # ], p=1),

                    A.OneOf([
                        A.HorizontalFlip(p=0.5),
                        A.Blur(p=0.5)
                    ], p=1),

                    A.OneOf([
                        A.RandomContrast(p=0.5, limit=(-0.5, 0.3)),
                        A.RandomBrightness(p=0.5, limit=(-0.2, 0.3)),
                    ], p=1)
                ])
                augmented_image = transform(image=image)['image']
                cv2.imwrite(output_path + '/' + label + '/' + file_name + '_' + str(i) + '_' + str(time.time()) + '.jpg', augmented_image)

        else:
            pass

    return output_path


if __name__ == "__main__":
    dataset_path = sys.argv[1]
    dataset_name = dataset_path.split('/')[-1]
    output_path = '/data/backup/pervinco_2020/Auged_datasets/' + dataset_name
    df = make_df(dataset_path)
    show_img_distribution(df)

    train_set, valid_set = train_test_split(df, test_size=0.2, shuffle=True)
    show_splited_datasets(train_set, valid_set)

    while True:
        print("Start Aug Process??? Press y or n")
        a = input()
        

        if a == 'y':
            print('How many augmentation do you want?')
            aug_num = float(input())

            output_train = aug_processing(train_set, output_path, int(aug_num), is_train=True)
            output_train_df = make_df(output_train)

            output_valid = aug_processing(valid_set, output_path, int(aug_num * 0.2), is_train=False)
            output_valid_df = make_df(output_valid)

            show_splited_datasets(output_train_df, output_valid_df)
            break


        elif a == 'n':
            break


        else:
            print("Please press y or n")
            continue


