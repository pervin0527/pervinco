import os
import random
from glob import glob
from shutil import copyfile

def generate_random_idx(maximum):
    idx_list = []
    a = random.randint(0, maximum)

    for i in range(maximum):
        while a in idx_list:
            a = random.randint(0, maximum)
        idx_list.append(a)

    return idx_list


def make_folder():
    if not os.path.isdir(outpath):
        os.makedirs(f"{outpath}/images")
        os.makedirs(f"{outpath}/annotations")


def generate_validset(random_list, image_list):
    for r in random_list:
        image_file = image_list[r]
        print(image_file)
        filename = image_file.split('/')[-1].split('.')[0]

        if os.path.isfile(f"{annotations_dir}/{filename}.xml"):
            copyfile(image_file, f"{outpath}/images/{filename}.jpg")
            copyfile(f"{annotations_dir}/{filename}.xml", f"{outpath}/annotations/{filename}.xml")


if __name__ == "__main__":
    ratio = 0.1
    outpath = "/data/Datasets/Seeds/SPC/set11/valid2"
    root = "/data/Datasets/Seeds/SPC/2021-11-24/videos"
    images_dir = f"{root}/images"
    annotations_dir = f"{root}/annotations"

    images = glob(f"{images_dir}/*.jpg")
    print(len(images))

    ri_list = generate_random_idx(int(len(images) * ratio))
    print(len(ri_list))

    make_folder()
    generate_validset(ri_list, images)

