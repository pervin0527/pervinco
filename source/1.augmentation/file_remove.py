import os
from glob import glob

root_path = "/data/Datasets/Seeds/SPC/set9"
annotation_path = f"{root_path}/images"
annotation_list = sorted(glob(f"{annotation_path}/*.jpg"))
print(len(annotation_list))

count = 0
for xml_file in annotation_list:
    filename = xml_file.split('/')[-1].split('.')[0]
    print(filename)

    if not os.path.isfile(f"{root_path}/annotations/{filename}.xml"):
        count += 1
        os.remove(f"{annotation_path}/{filename}.jpg")

print(count)