import os
import glob
from shutil import copyfile

def multiple_sets():
    ds_list = glob.glob(f"{ds_path}/*")
    print(len(ds_list))

    if not os.path.isdir(f"{'/'.join(outpath.split('/'))}/images") and not os.path.isdir(f"{'/'.join(outpath.split('/'))}/annotations"):
        os.makedirs(f"{'/'.join(outpath.split('/'))}/images")
        os.makedirs(f"{'/'.join(outpath.split('/'))}/annotations")

        print(f"{'/'.join(outpath.split('/'))}/images", f"{'/'.join(outpath.split('/'))}/annotations maded!")

    else:
        print("Already folders exist")

    for folder in ds_list:
        folder_name = folder.split('/')[-1]
        jpg_files = glob.glob(f'{folder}/*.jpg')
        
        for image in jpg_files:
            filename = image.split('/')[-1].split('.')[0]

            if os.path.isfile(f'{ds_path}/{folder_name}/{filename}.xml'):
                copyfile(f'{ds_path}/{folder_name}/{filename}.jpg', f'{outpath}/images/{filename}.jpg')
                copyfile(f'{ds_path}/{folder_name}/{filename}.xml', f'{outpath}/annotations/{filename}.xml')


def single_set():
    images_list = glob.glob(f"{ds_path}/*.jpg")

    for image_file in images_list:
        filename = image_file.split('/')[-1].split('.')[0]

        if os.path.isfile(f"{ds_path}/{filename}.xml"):
            copyfile(f'{ds_path}/{filename}.jpg', f'{outpath}/images/{filename}.jpg')
            copyfile(f'{ds_path}/{filename}.xml', f'{outpath}/annotations/{filename}.xml')


if __name__ == "__main__":
    # ds_path = "/data/Datasets/Seeds/SPC/2021-11-11/videos/frames"
    ds_path = "/data/Datasets/Seeds/SPC/set9/images"
    outpath = "/data/Datasets/Seeds/SPC/set9/train"
    
    single_set()