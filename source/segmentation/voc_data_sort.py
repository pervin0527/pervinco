import os
import shutil

def sorting_ds(file_list, save_path, is_train):
    if is_train:
        save_path = f"{save_path}/train"
    
    else:
        save_path = f"{save_path}/valid"

    if not os.path.isdir(save_path):
        os.makedirs(f"{save_path}/images")
        os.makedirs(f"{save_path}/annotations")
        os.makedirs(f"{save_path}/masks")

    f = open(file_list, 'r')
    while True:
        line = f.readline().replace('\n', '')
        if line is None or len(line) == 0:
            break
        
        image_file = f"{image_dir}/{line}.jpg"
        annotation_file = f"{annotation_dir}/{line}.png"
        mask_file = f"{mask_dir}/{line}.png"

        shutil.copyfile(image_file, f"{save_path}/images/{line}.jpg")
        shutil.copyfile(annotation_file, f"{save_path}/annotations/{line}.png")
        shutil.copyfile(mask_file, f"{save_path}/masks/{line}.png")

    f.close()


if __name__ == "__main__":
    root = "/data/Datasets/VOCdevkit/VOC2012"
    save_path = f"{root}/Segmentation"
    annotation_dir = f"{root}/SegmentationClass"
    image_dir = f"{root}/JPEGImages"
    mask_dir = f"{root}/SegmentationRaw"

    train_file_list = f"{root}/ImageSets/Segmentation/train.txt"
    valid_file_list = f"{root}/ImageSets/Segmentation/val.txt"
    
    sorting_ds(train_file_list, save_path, True)
    sorting_ds(valid_file_list, save_path, False)