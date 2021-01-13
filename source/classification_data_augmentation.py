import sys, glob, cv2, os, math, time, pathlib, datetime, argparse
import albumentations as A
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


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

    print(train_images, valid_images)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Classification dataset augmentation')
    parser.add_argument('--input_path', type=str)
    parser.add_argument('--num_of_aug', type=int, default=3000)
    args = parser.parse_args()

    seed_path = args.input_path
    aug_num = args.num_of_aug

    TODAY = datetime.datetime.now().strftime("%Y.%m.%d_%H:%M:%S")
    DATASET_NAME = seed_path.split('/')[-1]
    OUTPUT_PATH = seed_path.split('/')[:-2]
    OUTPUT_PATH = '/'.join(OUTPUT_PATH) + f'/Auged_datasets/{DATASET_NAME}_{TODAY}'

    label_list = sorted(os.listdir(seed_path + '/'))
    n_classes = len(label_list)

    visualize(seed_path, label_list)
    train_images, valid_images, train_info, valid_info = split_seed(seed_path)
    train_dataset_path = augmentation(train_images, True, train_info, aug_num)
    valid_dataset_path = augmentation(valid_images, False, valid_info, aug_num)
    aug_visualize(train_dataset_path, valid_dataset_path)