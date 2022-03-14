import os
import cv2
import random
import pandas as pd
import albumentations as A
from glob import glob
from tqdm import tqdm

ds_path = "/data/Datasets/SPC"
label_path = f"{ds_path}/Labels/labels.txt"
df = pd.read_csv(label_path, sep=',', index_col=False, header=None)
CLASSES = df[0].tolist()
print(CLASSES)

# transform = A.Compose([
#     A.Resize(width=320, height=320, p=1),
#     A.RandomBrightnessContrast(p=1)
# ])

# for label in CLASSES:
#     files = sorted(glob(f"{ds_path}/Cvat/{label}/*/JPEGImages/*"))

#     if not os.path.isdir(f"{ds_path}/Cls/train/{label}"):
#         os.makedirs(f"{ds_path}/Cls/train/{label}")
    
#     for idx in tqdm(range(len(files))):
#         file = files[idx]
#         file_name = file.split('/')[-1].split('.')[0]
#         image = cv2.imread(file)

#         transformed = transform(image=image)
#         transformed_image = transformed['image']
#         cv2.imwrite(f"{ds_path}/Cls/train/{label}/{file_name}.jpg", transformed_image)

#     files = random.choices(files, k=int(len(files) * 0.1))

#     if not os.path.isdir(f"{ds_path}/Cls/valid/{label}"):
#         os.makedirs(f"{ds_path}/Cls/valid/{label}")

#     for idx in tqdm(range(len(files))):
#         file = files[idx]
#         file_name = file.split('/')[-1].split('.')[0]
#         image = cv2.imread(file)

#         transformed = transform(image=image)
#         transformed_image = transformed['image']
#         cv2.imwrite(f"{ds_path}/Cls/valid/{label}/{file_name}.jpg", transformed_image)


if not os.path.isdir(f"{ds_path}/Cls/train/Background"):
    os.makedirs(f"{ds_path}/Cls/train/Background")
    os.makedirs(f"{ds_path}/Cls/valid/Background")

bg_files = sorted(glob(f"{ds_path}/download/*/*"))
print(len(bg_files))
for idx in tqdm(range(len(bg_files))):
    bg_file = bg_files[idx]

    try:
        name = bg_file.split('/')[-1].split('.')[0]

        image = cv2.imread(bg_file)
        image = cv2.resize(image, (320, 320))
        cv2.imwrite(f"{ds_path}/Cls/train/Background/{name}.jpg", image)
        cv2.imwrite(f"{ds_path}/Cls/valid/Background/{name}.jpg", image)

    except:
        print(bg_file)
    