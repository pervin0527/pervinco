from src.utils import read_label_file, read_xml, get_files

ROOT_PATH = "/data/Datasets/SPC"
LABEL_PATH = f"{ROOT_PATH}/Labels/labels.txt"

# frame_000004_0
IMAGE_PATH = f"{ROOT_PATH}/full-name-front/images"
ANNOT_PATH = f"{ROOT_PATH}/full-name-front/annotations"

classes = read_label_file(LABEL_PATH)

images, annotations = get_files(IMAGE_PATH), get_files(ANNOT_PATH)

max_ratio = 0
for (image, annot) in zip(images, annotations):
    bboxes, labels = read_xml(annot, classes, format='pascal_voc')

    for xmin, ymin, xmax, ymax in bboxes:
        bbox_width = xmax - xmin
        bbox_height = ymax - ymin

        bbox_ratio = bbox_width / bbox_height
        if bbox_ratio > max_ratio:
            max_ratio = bbox_ratio

print(max_ratio)