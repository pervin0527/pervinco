import cv2
import random
import albumentations as A
from tqdm import tqdm
from utils import make_file_list, make_save_dir, load_annot_data, annot_write

def make_seed_set(files):
    make_save_dir(save_dir)

    checker = set()
    random.shuffle(files)
    for idx in tqdm(range(len(files))):
        image_file, annot_file = files[idx]
        image = cv2.imread(image_file)
        bboxes, labels = load_annot_data(annot_file, target_classes=classes)

        for label in labels:
            if label[0] in classes:
                checker.add(label[0])
                try:
                    transformed = transform(image=image, bboxes=bboxes, labels=labels)
                    t_image, t_bboxes, t_labels = transformed["image"], transformed["bboxes"], transformed["labels"]

                    cv2.imwrite(f"{save_dir}/JPEGImages/{idx:>06}.jpg", t_image)
                    annot_write(f"{save_dir}/Annotations/{idx:06}.xml", t_bboxes, t_labels, t_image.shape[:2])
                    
                    draw_img = t_image.copy()
                    for bbox in t_bboxes:
                        xmin, ymin, xmax, ymax = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                        cv2.rectangle(draw_img, (xmin, ymin), (xmax, ymax), color=(0,0,255), thickness=2)
                    cv2.imwrite(f"{save_dir}/Results/{idx:>06}.jpg", draw_img)

                except:
                    print(image_file)
    print(checker)

    
if __name__ == "__main__":
    data_dir = ["/home/ubuntu/Datasets/BR/cvat/*",
                "/home/ubuntu/Datasets/SPC/Cvat/Baskin_robbins/*", 
                "/home/ubuntu/Datasets/BR/total"]
    save_dir = "/home/ubuntu/Datasets/BR/seed0_384"
    classes = ["Baskin_robbins"]

    img_size = 384
    transform = A.Compose([
        A.Resize(img_size, img_size, p=1),
    ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]))

    files = make_file_list(data_dir)
    make_seed_set(files)