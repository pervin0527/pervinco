import os
import shutil
from glob import glob
from tqdm import tqdm

if __name__ == "__main__":
    dir = "/home/ubuntu/Datasets/BR/frames"
    save_dir = "/home/ubuntu/Datasets/BR/total"

    if not os.path.isdir(f"{save_dir}/JPEGImages") and not os.path.isdir(f"{save_dir}/Annotations"):
        os.makedirs(f"{save_dir}/JPEGImages")
        os.makedirs(f"{save_dir}/Annotations")
        os.makedirs(f"{save_dir}/Results")

    total_imgs = sorted(glob(f"{dir}/*/*"))
    for idx in tqdm(range(len(total_imgs))):
        img_file = total_imgs[idx]
        shutil.copyfile(img_file, f"{save_dir}/JPEGImages/{idx:>09}.jpg")