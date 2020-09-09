import pandas as pd
import os
import shutil

main_label_file = '/data/backup/pervinco_2020/datasets/total_beverage/labels_beverage_final.txt'
df = pd.read_csv(main_label_file, sep = ' ', index_col=False, header=None)
CLASS_NAMES = df[0].tolist()
CLASS_NAMES = sorted(CLASS_NAMES)


images_path = '/data/backup/pervinco_2020/datasets/total_beverage/seed_images/d_seed_3'
label_list = sorted(os.listdir(images_path))

print(len(CLASS_NAMES), len(label_list))

out_list = []
for label in label_list:
    if label in CLASS_NAMES:
        pass

    else:
        out_list.append(label)
        # shutil.rmtree(images_path + '/' + label)

print(out_list)

no_list = []
for i in CLASS_NAMES:
    if i not in label_list:
        no_list.append(i)

    else:
        pass

print(no_list)
print(len(no_list))