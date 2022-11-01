import os
import cv2
from tqdm import tqdm
from glob import glob

if __name__ == "__main__":
    data_dir = "/data/Datasets/SPC/download"
    output_dir = "/data/Datasets/SPC/Background"
    size = 640

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    files = sorted(glob(f"{data_dir}/*/*.jpg"))
    for idx in tqdm(range(len(files))):
        image = cv2.imread(files[idx])
        image = cv2.resize(image, (size, size))

        cv2.imwrite(f"{output_dir}/{idx:>06}.jpg", image)