import cv2
from tqdm import tqdm
from src.utils import read_label_file, read_xml, visualize, get_files, write_xml, make_save_dir

data_path = "/data/Datasets/COCO2017"
image_path = f"{data_path}/images"
annotation_path = f"{data_path}/annotations"
label_path = f"{data_path}/Labels/labels.txt"
save_path = f"{data_path}/CUSTOM"

targets = ['person', 'car', 'motorbike', 'bus', 'truck', 'backpack', 'umbrella', 'bottle', 'cup', 'laptop', 'mouse', 'keyboard']

classes = read_label_file(label_path)
print(classes)
images, annotations = get_files(image_path), get_files(annotation_path)
print(len(images), len(annotations))

make_save_dir(save_path)
for index in tqdm(range(len(images))):
    filename = images[index].split('/')[-1].split('.')[0]
    image = cv2.imread(images[index])
    image_height, image_width = image.shape[:-1]

    bboxes, labels = read_xml(annotations[index], classes, format='pascal_voc')
    # print(labels)

    for label in labels:
        if not label in targets:
            del bboxes[labels.index(label)]
            labels.remove(label)

    write_xml(f"{save_path}/annotations", bboxes, labels, filename, image_height, image_width, format="pascal_voc")
    cv2.imwrite(f"{save_path}/images/{filename}.jpg", image)
    # visualize(image, bboxes, labels, format="pascal_voc", show_info=True)