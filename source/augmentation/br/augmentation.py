import os
import cv2
import albumentations as A
from glob import glob
from tqdm import tqdm
from utils import make_file_list, read_xml, read_label_file, visualize, make_save_dir, write_xml

def augmentation(total_files):
    make_save_dir(save_dir)
    for idx in tqdm(range(len(total_files))):
        img_file, annot_file = total_files[idx]
        image = cv2.imread(img_file)
        bboxes, labels = read_xml(annot_file, classes, format="pascal_voc")

        try:
            transformed = transform(image=image, bboxes=bboxes, labels=labels)
            t_image, t_bboxes, t_labels = transformed["image"], transformed["bboxes"], transformed["labels"]
            # visualize(t_image, t_bboxes, t_labels, format="pascal_voc")

            result = t_image.copy()
            for bbox in t_bboxes:
                cv2.rectangle(result, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255))

            cv2.imwrite(f"{save_dir}/result/{idx:>06}.jpg", result)
            cv2.imwrite(f"{save_dir}/images/{idx:>06}.jpg", t_image)
            write_xml(f"{save_dir}/annotations", t_bboxes, t_labels, f"{idx:>06}", img_size, img_size, format="pascal_voc")
        
        except:
            print(f"{img_file}, {annot_file}")

if __name__ == "__main__":
    dataset1_dir = "/home/ubuntu/Datasets/SPC/Cvat/Baskin_robbins"
    dataset2_dir = "/home/ubuntu/Datasets/BR/cvat"
    label_dir = "/home/ubuntu/Datasets/BR/Labels/labels.txt"
    save_dir = "/home/ubuntu/Datasets/BR/set2"

    img_size = 384
    transform = A.Compose([
        A.Resize(img_size, img_size, p=1)
    ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]))

    classes = read_label_file(label_dir)
    print(classes)

    image_files1, annot_files1 = make_file_list(dataset1_dir)
    image_files2, annot_files2 = make_file_list(dataset2_dir)
    print(len(image_files1), len(annot_files1))
    print(len(image_files2), len(annot_files2))

    dataset = list(zip(image_files1, annot_files1))
    dataset.extend(list(zip(image_files2, annot_files2)))
    augmentation(dataset)