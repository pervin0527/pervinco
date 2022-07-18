import os
import cv2
import albumentations as A

def make_label_field(n):
    labels = []
    for _ in range(n):
        labels.extend(['face'])

    return labels


def refine_data(txt_file):
    f = open(txt_file, "r")
    total_annot = f.readlines()

    total_data = {}
    while total_annot:
        line = total_annot.pop(0).strip().split()
        img_path = line[206]
        bbox = list(map(int, line[196:200]))

        if img_path not in total_data.keys():
            total_data.update({img_path:[bbox]})

        else:
            total_data[img_path].append(bbox)

    return total_data


def write_txt(data_dict, save_dir, is_train):
    if is_train:
        save_dir = f"{save_dir}/train"
    
    else:
        save_dir = f"{save_dir}/test"
        
    if not os.path.isdir(save_dir):
        os.makedirs(f"{save_dir}/images")

    f = open(f"{save_dir}/list.txt", "w")
    for idx, data in enumerate(list(data_dict.items())):
        file, bboxes = data[0], data[1]
        labels = make_label_field(len(bboxes))
        print(file)

        image = cv2.imread(f"{image_dir}/{file}")
        transformed = transform(image=image, bboxes=bboxes, labels=labels)
        t_image, t_bboxes = transformed['image'], transformed['bboxes']

        f.write(f"{save_dir}/images/{idx}.jpg")
        cv2.imwrite(f"{save_dir}/images/{idx}.jpg", t_image)

        for bbox in t_bboxes:
            f.write(' ')
            xmin, ymin, xmax, ymax = bbox
            label = classes.index("face")
            f.write(f"{int(xmin)},{int(ymin)},{int(xmax)},{int(ymax)},{int(label)}")
        f.write('\n')


if __name__ == "__main__":
    root_dir = "/data/Datasets/WFLW"
    image_dir = f"{root_dir}/WFLW_images"
    train_txt = f"{root_dir}/WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_train.txt"
    test_txt = f"{root_dir}/WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_test.txt"
    save_dir = f"{root_dir}/CUSTOM_TXT"
    classes = ["face"]

    transform = A.Compose([
        A.Resize(512, 512, always_apply=True)
    ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=['labels']))

    # sample_txt = "./sample.txt"
    train_data = refine_data(train_txt)
    write_txt(train_data, save_dir, True)

    test_data = refine_data(test_txt)
    write_txt(test_data, save_dir, False)