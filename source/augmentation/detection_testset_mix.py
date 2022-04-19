import shutil
import numpy as np
from src.utils import get_files, make_save_dir

def make_file_list(ds_list):
    ds_dict = {}
    
    for idx, path in enumerate(ds_list):
        images, annotations = get_files(f"{path}/images"), get_files(f"{path}/annotations")
        dataset = list(zip(images, annotations))

        ds_dict.update({idx : dataset})

    return ds_dict

if __name__ == "__main__":
    datasets = [
        "/data/Datasets/SPC/Testset/Real",
        "/data/Datasets/SPC/Testset/day_night"
    ]
    save_dir = "/data/Datasets/SPC/Testset/test3"
    ratio = [.3, .7]

    assert len(datasets) == len(ratio), "len dataset and len ratio must be same"
    
    items = [idx for idx in range(len(datasets))]
    nums = np.random.choice(items, size=100, p=ratio)
    totalset = make_file_list(datasets)

    print(len(nums))
    make_save_dir(save_dir)
    for num in nums:
        files = totalset.get(num)
        file = files.pop(np.random.randint(0, len(files)))

        filename = file[0].split('/')[-1].split(('.'))[0]
        shutil.copyfile(file[0], f'{save_dir}/images/{num}_{filename}.jpg')
        shutil.copyfile(file[1], f'{save_dir}/annotations/{num}_{filename}.xml')