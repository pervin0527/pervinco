import cv2
import albumentations as A
from glob import glob
from tqdm import tqdm
from utils import make_file_list, make_save_dir, load_annot_data, annot_write

def make_seed_set(files):
    make_save_dir(save_dir)

    for idx in tqdm(range(len(files))):
        image_file, annot_file = files[idx]
        image = cv2.imread(image_file)
        bboxes, labels = load_annot_data(annot_file)

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

if __name__ == "__main__":
    data_dir = ["/data/Datasets/BR/cvat/*", "/data/Datasets/SPC/Cvat/Baskin_robbins/*"]
    save_dir = "/data/Datasets/BR/Seeds"

    img_size = 640
    transform = A.Compose([
        A.Resize(img_size, img_size, p=1),
    ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]))

    files = make_file_list(data_dir)
    make_seed_set(files)