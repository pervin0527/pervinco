import os
import cv2
from tqdm import tqdm
from src.utils import read_label_file, read_xml, get_files, get_content_filename, write_xml, visualize, make_save_dir


def search(root_dir):
    folders = []

    contents = os.listdir(root_dir)
    for content in contents:
        if os.path.isdir(f"{ROOT}/{content}") and content != f"{SAVE_FOLDER}":
            folders.append(content)

    return sorted(folders)


def process(img_dir, annot_dir, number):
    images, annotations = get_files(img_dir), get_files(annot_dir)
    # print(len(images), len(annotations))

    for index in tqdm(range(len(images))):
        image, annot = images[index], annotations[index]
        img_filename = image.split('/')[-1].split('.')[0]

        image = cv2.imread(image)

        img_height, img_width = image.shape[:-1]
        if img_filename == get_content_filename(annot):
            bboxes, labels = read_xml(annot, LABELS, format='pascal_voc')

            if index == 0:
                visualize(image, bboxes, labels, format='pascal_voc', show_info=False)
            
            cv2.imwrite(f"{ROOT}/{SAVE_FOLDER}/images/{img_filename}_{number}.jpg", image)
            write_xml(f"{ROOT}/{SAVE_FOLDER}/annotations", bboxes, labels, img_filename, img_height, img_width, 'pascal_voc')
        
    
if __name__ == "__main__":
    ROOT = "/data/Datasets/SPC/Cvat/full-name-seed1"
    LABEL_DIR = "/data/Datasets/SPC/Labels/labels.txt"
    LABELS = read_label_file(LABEL_DIR)
    SAVE_FOLDER = "test"
    target_folders = search(ROOT)
    print(target_folders)

    make_save_dir(f"{ROOT}/{SAVE_FOLDER}")

    for idx, folder in enumerate(target_folders):
        image_dir = f"{ROOT}/{folder}/JPEGImages"
        annot_dir = f"{ROOT}/{folder}/Annotations"

        process(image_dir, annot_dir, idx)

    images, annotations = get_files(f"{ROOT}/{SAVE_FOLDER}/images"), get_files(f"{ROOT}/{SAVE_FOLDER}/annotations")
    image = cv2.imread(images[0])
    bboxes, labels = read_xml(annotations[0], LABELS)
    visualize(image, bboxes, labels, "pascal_voc")