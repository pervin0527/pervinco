import os
import glob
from shutil import copyfile

ds_path = "/data/Datasets/Seeds/DMC/frames"
ds_list = glob.glob(f"{ds_path}/*.jpg")
print(len(ds_list))

if not os.path.isdir(f"{'/'.join(ds_path.split('/')[:-1])}/images") and not os.path.isdir(f"{'/'.join(ds_path.split('/')[:-1])}/annotations"):
    os.makedirs(f"{'/'.join(ds_path.split('/')[:-1])}/images")
    os.makedirs(f"{'/'.join(ds_path.split('/')[:-1])}/annotations")

    print(f"{'/'.join(ds_path.split('/')[:-1])}/images", f"{'/'.join(ds_path.split('/')[:-1])}/annotations maded!")

else:
    print("Already folders exist")

for jpg_file in ds_list:
    jpg_filename = (jpg_file.split('/')[-1]).split('.')[0]
    print(jpg_filename)

    if os.path.isfile(f"{ds_path}/{jpg_filename}.xml"):
        copyfile(f"{ds_path}/{jpg_filename}.xml", f"{'/'.join(ds_path.split('/')[:-1])}/annotations/{jpg_filename}.xml")
        copyfile(f"{ds_path}/{jpg_filename}.jpg", f"{'/'.join(ds_path.split('/')[:-1])}/images/{jpg_filename}.jpg")

print("Done")