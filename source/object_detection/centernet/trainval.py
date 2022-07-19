import os
import cv2
import albumentations as A
import xml.etree.ElementTree as ET
from collections import deque
from lxml.etree import Element, SubElement

def write_xml(save_path, bboxes, labels, filename, height, width):
    root = Element("annotation")
    
    folder = SubElement(root, "folder")
    folder.text = "images"

    file_name = SubElement(root, "filename")
    file_name.text = f'{filename}.jpg'
    
    size = SubElement(root, "size")
    w = SubElement(size, "width")
    w.text = str(width)
    h = SubElement(size, "height")
    h.text = str(height)
    depth = SubElement(size, "depth")
    depth.text = "3"

    if labels:
        for label, bbox in zip(labels, bboxes):
            obj = SubElement(root, 'object')
            name = SubElement(obj, 'name')
            name.text = label
            pose = SubElement(obj, 'pose')
            pose.text = 'Unspecified'
            truncated = SubElement(obj, 'truncated')
            truncated.text = '0'
            difficult = SubElement(obj, 'difficult')
            difficult.text = '0'
            bndbox = SubElement(obj, 'bndbox')
            xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3]

            node_xmin = SubElement(bndbox, 'xmin')
            node_xmin.text = str(int(xmin))
            node_ymin = SubElement(bndbox, 'ymin')
            node_ymin.text = str(int(ymin))
            node_xmax = SubElement(bndbox, 'xmax')
            node_xmax.text = str(int(xmax))
            node_ymax = SubElement(bndbox, 'ymax')
            node_ymax.text = str(int(ymax))
    
    tree = ET.ElementTree(root)    
    tree.write(f"{save_path}/{filename}.xml")


def augmentation(image, bboxes, labels):
    transformed = transform(image=image, bboxes=bboxes, labels=labels)
    t_image, t_bboxes, t_labels = transformed['image'], transformed['bboxes'], transformed['labels']

    return t_image, t_bboxes, t_labels


def refine_coordinates(bbox):
    x, y, w, h = bbox
    xmin = x
    ymin = y
    xmax = w + xmin
    ymax = h + ymin

    return [xmin, ymin, xmax, ymax]


def make_label_field(n):
    labels = []
    for _ in range(n):
        labels.extend(['face'])

    return labels


def get_wider_dataset(txt_list, img_dir_list):
    dataset = []

    if not os.path.isdir(WIDER_SAVE_DIR):
        os.makedirs(f"{WIDER_SAVE_DIR}/images")

    for txt_file, img_dir in zip(txt_list, img_dir_list):
        file = open(txt_file, "r")
        lines = deque(file.readlines())

        index = 0
        while lines:
            tmp = []
            img_path = lines.popleft()[:-1]
            n_boxes = int(lines.popleft()[:-1])
            print(img_path)

            if 0 < n_boxes <= MAX_OBJECTS:
                bboxes = []
                for _ in range(int(n_boxes)):
                    bbox_with_attr = lines.popleft()[:-1].split()
                    bbox = list(map(int, bbox_with_attr[:4]))
                    bbox = refine_coordinates(bbox)
                    bboxes.append(bbox)

                image = cv2.imread(f"{img_dir}/images/{img_path}")
                classes = make_label_field(int(n_boxes))
                try:
                    augment_image, augment_bboxes, augment_classes = augmentation(image, bboxes, classes)
                    
                    file_name = f"{index:>06}.jpg"
                    cv2.imwrite(f"{WIDER_SAVE_DIR}/images/{file_name}", augment_image)

                    tmp.append(f"{WIDER_SAVE_DIR}/images/{file_name}")
                    for (bbox, label) in zip(augment_bboxes, augment_classes):
                        xmin, ymin, xmax, ymax = bbox
                        c = LABELS.index(label)
                        tmp.append([xmin, ymin, xmax, ymax, c])

                    dataset.append(tmp)
                    index += 1

                except:
                    pass

            elif n_boxes > MAX_OBJECTS:
                for _ in range(n_boxes):
                    lines.popleft()[:-1].split()

            elif n_boxes == 0:
                lines.popleft()[:-1].split()

    return dataset


def get_wflw_data(txt_file):
    total = {}
    f = open(txt_file, "r")
    lines = f.readlines()

    while lines:
        line = lines.pop(0).strip().split()
        img_path = line[206]
        bbox = list(map(int, line[196:200]))

        if img_path not in total.keys():
            total.update({img_path:[bbox]})

        else:
            total[img_path].append(bbox)
    
    dataset = []

    if not os.path.isdir(WFLW_SAVE_DIR):
        os.makedirs(f"{WFLW_SAVE_DIR}/images")

    for index, data in enumerate(list(total.items())):
        file, bboxes = data[0], data[1]
        print(file, len(bboxes))
        labels = make_label_field(len(bboxes))
        image = cv2.imread(f"{WFLW_IMAGES}/{file}")

        tmp = []
        try:
            file_name = f"{index:>06}.jpg"
            augment_image, augment_bboxes, augment_classes = augmentation(image, bboxes, labels)
            cv2.imwrite(f"{WFLW_SAVE_DIR}/images/{file_name}", augment_image)
            tmp.append(f"{WFLW_SAVE_DIR}/images/{file_name}")

            for (bbox, label) in zip(augment_bboxes, augment_classes):
                xmin, ymin, xmax, ymax = bbox
                c = LABELS.index(label)
                tmp.append([xmin, ymin, xmax, ymax, c])

            dataset.append(tmp)

        except:
            pass

    return dataset


def record_dataset(dataset, save_dir, format="txt"):
    record = open(f"{save_dir}/list.txt", "w")

    for data in dataset:
        image_path = data[0]
        record.write(image_path)

        for bbox in data[1:]:
            record.write(" ")
            xmin, ymin, xmax, ymax = bbox[0:4]
            label = bbox[-1]
            record.write(f"{int(xmin)},{int(ymin)},{int(xmax)},{int(ymax)},{int(label)}")

        record.write("\n")

    if format == "xml":
        if not os.path.isdir(f"{save_dir}/annotations"):
            os.makedirs(f"{save_dir}/annotations")

        for data in dataset:
            image_path = data.pop(0)
            file_name = image_path.split('/')[-1].split('.')[0]

            bboxes, labels = [], []
            for bbox in data:
                xmin, ymin, xmax, ymax = bbox[0:4]
                label = bbox[-1]

                bboxes.append([int(xmin), int(ymin), int(xmax), int(ymax)])
                labels.append(str(LABELS[label]))

            write_xml(f"{save_dir}/annotations", bboxes, labels, file_name, 512, 512)


if __name__ == "__main__":
    WIDER_IMAGES = ["/data/Datasets/WIDER/WIDER_train",
                    "/data/Datasets/WIDER/WIDER_val"]
    WIDER_TXT = ["/data/Datasets/WIDER/wider_face_split/wider_face_train_bbx_gt.txt",
                 "/data/Datasets/WIDER/wider_face_split/wider_face_val_bbx_gt.txt"]

    WFLW_IMAGES = "/data/Datasets/WFLW/WFLW_images"
    WFLW_TXT = "/data/Datasets/WFLW/WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_test.txt"

    WIDER_SAVE_DIR = "/data/Datasets/FACE_DETECTION/train"
    WFLW_SAVE_DIR = "/data/Datasets/FACE_DETECTION/test"
    LABELS = ["face"]
    MAX_OBJECTS = 4
    IMG_SIZE = 384
    FORMAT = "xml"

    transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE, always_apply=True)
    ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]))

    train_dataset = get_wider_dataset(WIDER_TXT, WIDER_IMAGES)
    record_dataset(train_dataset, WIDER_SAVE_DIR, format=FORMAT)
    
    test_dataset = get_wflw_data(WFLW_TXT)
    record_dataset(test_dataset, WFLW_SAVE_DIR, format=FORMAT)