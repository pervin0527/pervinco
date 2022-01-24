import os
import cv2
import albumentations as A
from glob import glob
from tqdm import tqdm
from src.utils import write_xml

if __name__ == "__main__":
    background_dir = "/data/Datasets/SPC/Background"
    IMG_SIZE = 384

    if not os.path.isdir(f"{background_dir}/annotations"):
        os.makedirs(f"{background_dir}/annotations")

    images = glob(f"{background_dir}/images/*")
    for idx in tqdm(range(len(images))):
        file = images[idx]
        file_name = file.split('/')[-1].split('.')[0]

        image = cv2.imread(file)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        write_xml(f"{background_dir}/annotations", None, None, file_name, IMG_SIZE, IMG_SIZE, format='pascal_voc')