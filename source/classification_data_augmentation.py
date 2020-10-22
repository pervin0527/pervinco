import sys
import glob
import cv2
import os
import math
import pandas as pd
import seaborn as sns
import albumentations as A
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


def visualize(labels, files):
    x, y = labels, files

    plt.figure(figsize=(20, 10))
    plt.bar(x, y, width=0.9,)
    plt.xticks(x, rotation=90)
    plt.show()


def aug_visualize(labels, train_data_path, valid_data_path):
    train_images = []
    valid_images = []

    for label in labels:
        train_image_num = len(glob.glob(train_data_path + '/' + label + '/*.jpg'))
        valid_image_num = len(glob.glob(valid_data_path + '/' + label + '/*.jpg'))
        train_images.append(train_image_num)
        valid_images.append(valid_image_num)

    plt.figure(figsize=(20, 10))
    plt.bar(labels, train_images, width=0.9,)
    plt.bar(labels, valid_images, width=0.9,)
    plt.xticks(labels, rotation=90)
    plt.show()    


def make_df(path, label_list):
    result = []
    idx = 0
    file_num = []

    for label in label_list:
        file_list = glob.glob(os.path.join(path,label,'*'))
        file_num.append(len(file_list))
        
        for file in file_list:
            result.append([idx, label, file])
            idx += 1
            
    img_df = pd.DataFrame(result, columns=['idx','label','image_path'])
    visualize(label_list, file_num)
    return img_df


def split_process(df, is_train, output_path):
    path = df['image_path'].sort_index()

    if is_train:
        output_path = output_path + '/train'

    else:
        output_path = output_path + '/valid'

    for i in path:
        file_name = i.split('/')[-1]
        label = i.split('/')[-2]
        print(label, file_name)

        if not(os.path.isdir(output_path + '/' + label)):
            os.makedirs(output_path + '/' + label)

        else:
            pass

        image = cv2.imread(i)
        transform = A.Resize(224, 224)
        augmented_image = transform(image=image)['image']
        cv2.imwrite(output_path + '/' + label + '/' + file_name, augmented_image)

    return output_path


def augmentation(data_path, label_list, output_path, aug_num):
    is_train = data_path.split('/')[-1]

    if is_train == 'train':
        output_path = output_path + '/train'

    else:
        output_path = output_path + '/valid'

    for label in label_list:
        images = glob.glob(data_path + '/' + label + '/*.jpg')
        cnt = int(math.ceil(aug_num / len(images)))

        for img in images:
            auged_images = len(glob.glob(output_path + '/' + label + '/*.jpg'))
            if auged_images <= aug_num:
                print(label, img)
                file_name = img.split('/')[-1]
                file_name = file_name.split('.')[0]
                image = cv2.imread(img)

                for i in range(cnt):
                    transform = A.Compose([
                        A.Resize(224, 224, p=1),
                        A.HorizontalFlip(p=0.3),
                        A.VerticalFlip(p=0.1),
                        A.Blur(p=0.1),

                        # A.RandomRotate90(p=0.3),

                        A.OneOf([
                            A.RandomContrast(p=0.5, limit=(-0.5, 0.3)),
                            A.RandomBrightness(p=0.5, limit=(-0.2, 0.3)),
                        ], p=0.5)
                    ])
                    augmented_image = transform(image=image)['image']
                    cv2.imwrite(output_path + '/' + label + '/' + file_name + '_' + str(i) +'.jpg', augmented_image)

            else:
                pass

    return output_path


if __name__ == "__main__":
    seed_path = sys.argv[1]
    aug_num = int(sys.argv[2])
    dataset_name = seed_path.split('/')[-1]
    output_path = '/data/backup/pervinco_2020/Auged_datasets/' + dataset_name

    label_list = sorted(os.listdir(seed_path))
    seed_df = make_df(seed_path, label_list)
    print(len(label_list))

    train_set, valid_set = train_test_split(seed_df, test_size=0.2, shuffle=True)
    train_path = split_process(train_set, True, output_path)
    valid_path = split_process(valid_set, False, output_path)

    print(train_path, valid_path)
    train_data_path = augmentation(train_path, label_list, output_path, aug_num)
    valid_data_path = augmentation(valid_path, label_list, output_path, int(aug_num * 0.2))
    os.system('clear')
    print("Done")
    aug_visualize(label_list, train_path, valid_path)