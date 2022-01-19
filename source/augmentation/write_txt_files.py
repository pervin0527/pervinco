import os
from glob import glob

root = "/home/ubuntu/Datasets/SPC/full-name6"
target_paths = [f"{root}/train3", f"{root}/valid3"]

for folder in target_paths:
    images = sorted(glob(f"{folder}/v4-images/*.jpg"))

    f = open(f"{folder}/files.txt", 'w')
    for image in images:
        data = image + '\n'
        f.write(data)

f.close()