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


def build_dataset(files, mode):
    if mode == "train":
        output_dir = f"{save_path}/train"
    else:
        output_dir = f"{save_path}/valid"

    if not os.path.isdir(output_dir):
        os.makedirs(f"{output_dir}/images")
        os.makedirs(f"{output_dir}/masks")
        os.makedirs(f"{output_dir}/annotations")

    for file in files:
        image_file = f"{image_dir}/{file}.jpg"
        annotation_file = f"{annotation_dir}/{file}.png"
        mask_file = f"{mask_dir}/{file}.png"

        shutil.copyfile(image_file, f"{output_dir}/images/{file}.jpg")
        shutil.copyfile(annotation_file, f"{output_dir}/annotations/{file}.png")
        shutil.copyfile(mask_file, f"{output_dir}/masks/{file}.png")

def split_ds(train_list, valid_list):
    total_list = []

    for txt_file in [train_list, valid_list]:
        f = open(txt_file, 'r')
        while True:
            line = f.readline().replace('\n', '')
            if line is None or len(line) == 0:
                break
            total_list.append(line)

        f.close()
    
    build_dataset(total_list[:-num_valid], "train")
    build_dataset(total_list[-num_valid:], "valid")
    

if __name__ == "__main__":
    root = "/data/Datasets/VOCdevkit/VOC2012"
    save_path = f"{root}/BASIC"

    annotation_dir = f"{root}/SegmentationClass"
    image_dir = f"{root}/JPEGImages"
    mask_dir = f"{root}/SegmentationRaw"

    train_file_list = f"{root}/ImageSets/Segmentation/train.txt"
    valid_file_list = f"{root}/ImageSets/Segmentation/val.txt"
    
    # sorting_ds(train_file_list, save_path, True)
    # sorting_ds(valid_file_list, save_path, False)

    num_valid = 100
    split_ds(train_file_list, valid_file_list)