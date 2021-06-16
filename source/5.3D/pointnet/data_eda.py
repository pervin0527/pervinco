import os
import pathlib
from matplotlib import pyplot as plt

def visualize(path, label_list):
    images = []
    for label in label_list:
        files = os.listdir(f'{path}/{label}')
        file_num = len(files)

        images.append(file_num)

    x, y = label_list, images
    plt.figure(figsize=(10, 10))
    plt.bar(x, y, width=0.9,)
    plt.xticks(x, rotation=270)
    plt.show()

dataset_path = '/data/datasets/modelnet40_normal_resampled'
ds_path = pathlib.Path(dataset_path)

images = list(ds_path.glob('*/*.txt'))
images = [str(path) for path in images]
print(len(images))

labels = sorted(item.name for item in ds_path.glob('*/') if item.is_dir())
print(labels)

visualize(dataset_path, labels)