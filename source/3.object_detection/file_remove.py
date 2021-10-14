import os
from glob import glob

root_path = "/data/Datasets/Seeds/DMC/set4"
annotation_path = f"{root_path}/annotations"
annotation_list = sorted(glob(f"{annotation_path}/*.xml"))
print(len(annotation_list))

count = 0
for xml_file in annotation_list:
    filename = xml_file.split('/')[-1].split('.')[0]
    print(filename)

    if not os.path.isfile(f"{root_path}/images/{filename}.jpg"):
        count += 1
        os.remove(f"{annotation_path}/{filename}.xml")

print(count)