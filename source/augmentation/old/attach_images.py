import sys, glob, cv2, os, math, time, pathlib, datetime, argparse
import albumentations as A
import numpy as np
from random import randrange, choice, sample
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def visualize(path, label_list):
    images = []
    for label in label_list:
        files = os.listdir(path + '/' + label)
        file_num = len(files)

        images.append(file_num)

    x, y = label_list, images
    plt.figure(figsize=(10, 10))
    plt.bar(x, y, width=0.9,)
    plt.xticks(x, rotation=90)
    plt.show()
        

def aug_visualize(train_data_path, valid_data_path):
    train_images = []
    valid_images = []

    for label in label_list:
        train_image_num = len(os.listdir(train_data_path + '/' + label))
        valid_image_num = len(os.listdir(valid_data_path + '/' + label))

        train_images.append(train_image_num)
        valid_images.append(valid_image_num)

    print(train_images)
    print(valid_images)
    
    plt.figure(figsize=(10, 10))
    plt.bar(label_list, train_images, width=0.9,)
    plt.bar(label_list, valid_images, width=0.9,)
    plt.xticks(label_list, rotation=90)
    plt.show() 
    

def make_info(images):
    info = {}
    cnt = [0] * len(label_list)

    for image in images:
        label = image.split('/')[-2]
        
        if label in label_list:
            cnt[label_list.index(label)] += 1

    for label, num in zip(label_list, cnt):
        info.update({label:num})

    return info


def split_seed(ds_path):
    ds_path = pathlib.Path(ds_path)
    images = list(ds_path.glob('*/*'))
    images = [str(path) for path in images]

    train_images, valid_images = train_test_split(images, test_size=0.2)
    train_info = make_info(train_images)
    valid_info = make_info(valid_images)

    return sorted(train_images), sorted(valid_images), train_info, valid_info
    

def augmentation(images, is_train, info, aug_num):
    if is_train:
        output_path = f'{OUTPUT_PATH}/train'

    else:
        output_path = f'{OUTPUT_PATH}/valid'
        aug_num = int(aug_num * 0.2)

    for label in label_list:
        if not os.path.isdir(f'{output_path}/{label}'):
            os.makedirs(f'{output_path}/{label}')

    for i in tqdm(range(len(images))):
        image = images[i]
        image_name = image.split('/')[-1]
        label = image.split('/')[-2]

        cnt = int(math.ceil(aug_num / info[label]))
        total_images = len(os.listdir(f'{output_path}/{label}'))
        if total_images <= aug_num:
            image = cv2.imread(image)
            transform = A.Resize(224, 224)
            augmented_image = transform(image = image)['image']
            cv2.imwrite(f'{output_path}/{label}/orig_{image_name}', augmented_image)

            for c in range(cnt):
                transform = A.Compose([
                    A.Resize(224, 224, p=1),
                    A.HorizontalFlip(p=0.4),
                    A.VerticalFlip(p=0.3),
                    A.Blur(p=0.1),

                    A.OneOf([
                        A.RandomContrast(p=0.5, limit=(-0.5, 0.3)),
                        A.RandomBrightness(p=0.5, limit=(-0.2, 0.3))
                    ], p=0.5)
                ])
                augmented_image =transform(image=image)['image']
                cv2.imwrite(f'{output_path}/{label}/aug{c}_{image_name}', augmented_image)

    return output_path 


def get_foreground(fg_path):
    foreground = pathlib.Path(fg_path)
    classes = (list(foreground.glob('*')))
    classes = sorted([str(label).split('/')[-1] for label in classes])

    fg_list = []
    for label in classes:
        images = list(foreground.glob(label + '/*'))
        images = [str(image) for image in images]
        img_list = sample(images, int(aug_num * 0.01))
        fg_list += img_list

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
    for i in tqdm(range(len(fg_list))):
        idx = 0
        RAND_SIZE = randrange(100, 130)
        transform = A.Resize(IMG_SIZE, RAND_SIZE)
        x = randrange(0, 120)
        y = randrange(50, 70)

        # x = randrange(0, 120)
        # y = randrange(0, 50)

        fg_image = fg_list[i]
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
            
            if not os.path.isdir(f"{OUTPUT_PATH}/train/{fg_label}"):
                os.makedirs(f"{OUTPUT_PATH}/train/{fg_label}")
            cv2.imwrite(f"{OUTPUT_PATH}/train/{fg_label}/overlay_{idx}_{time.time()}.jpg", bg_image)

            # cv2.imshow("test", bg_image)
            # cv2.waitKey(0)

            idx += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Classification dataset augmentation')
    parser.add_argument('--input_path', type=str)
    parser.add_argument('--num_of_aug', type=int, default=3000)
    parser.add_argument('--overlay', type=str2bool, default=False)
    parser.add_argument('--background_path', required='--overlay' in sys.argv)
    args = parser.parse_args()

    seed_path = args.input_path
    aug_num = args.num_of_aug

    TODAY = datetime.datetime.now().strftime("%Y.%m.%d_%H:%M:%S")
    DATASET_NAME = seed_path.split('/')[-1]
    OUTPUT_PATH = seed_path.split('/')[:-2]
    OUTPUT_PATH = '/'.join(OUTPUT_PATH) + f'/Auged_datasets/{DATASET_NAME}/{TODAY}'

    label_list = sorted(os.listdir(seed_path + '/'))
    n_classes = len(label_list)

    visualize(seed_path, label_list)
    train_images, valid_images, train_info, valid_info = split_seed(seed_path)
    train_dataset_path = augmentation(train_images, True, train_info, aug_num)
    valid_dataset_path = augmentation(valid_images, False, valid_info, aug_num)

    if args.overlay == True:
        IMG_SIZE = 224
        fg_list = get_foreground(seed_path)
        bg_path = args.background_path

        overlay(fg_list)

    aug_visualize(train_dataset_path, valid_dataset_path)