import os
import cv2
from glob import glob
from src.utils import get_files

bg_path = "/data/Datasets/SPC/Seeds/Background2"
bg_folders = sorted(glob(f"{bg_path}/*"))
print(bg_folders)

if not os.path.isdir(f"{bg_path}/total"):
    os.makedirs(f"{bg_path}/total")

for idx1, folder in enumerate(bg_folders):
    files = get_files(folder)
    print(files)

    for idx2, file in enumerate(files):
        image = cv2.imread(file)
        image = cv2.resize(image, (512, 512))
        cv2.imwrite(f"{bg_path}/total/bg_{idx1}_{idx2}.jpg", image)