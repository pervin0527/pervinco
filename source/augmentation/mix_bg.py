import cv2
import random
import albumentations as A
from tqdm import tqdm
from src.data import Dataset, Augmentations
from src.custom_aug import MixUp, CutMix, Mosaic
from src.utils import read_label_file, read_xml, get_files, make_save_dir, write_xml, visualize

if __name__ == "__main__":
    AUG_N = 4
    ROOT_DIR = "/data/Datasets/SPC/full-name1"
    LABEL_DIR = "/data/Datasets/SPC/Labels/labels.txt"
    SAVE_DIR = "/data/Datasets/SPC/full-name1/augmentations"

    IMG_DIR = f"{ROOT_DIR}/images"
    ANNOT_DIR = f"{ROOT_DIR}/annotations"

    INCLUDE_BG = True
    BG_RATIO = 0.05
    BG_DIR = ["/data/Datasets/COCO2017", "/data/Datasets/SPC/Seeds/Background"]

    make_save_dir(SAVE_DIR)
    images, annotations = get_files(IMG_DIR), get_files(ANNOT_DIR)
    classes = read_label_file(LABEL_DIR)

    if INCLUDE_BG:
        for bg in BG_DIR:
            bg_images = get_files(f"{bg}/images")
            bg_images = random.sample(bg_images, int(len(images) * BG_RATIO))
            print(len(bg_images))

            for bg_img in bg_images:
                filename = bg_img.split('/')[-1].split('.')[0]
                image = cv2.imread(bg_img)
                height, width = image.shape[:-1]
                cv2.imwrite(f"{SAVE_DIR}/images/bg_{filename}.jpg", image)
                write_xml(f"{SAVE_DIR}/annotations", None, None, f"bg_{filename}", height, width, 'pascal_voc')

    dataset = Dataset(images, annotations, classes)
    # print(dataset[0])

    transform = A.Compose([
    A.OneOf([
        A.Sequential([
            A.Rotate(limit=5, p=1, border_mode=0),
            MixUp(dataset, rate_range=(0, 0.1), mix_label=False, p=0.5),
            A.RandomBrightnessContrast(p=1),
            A.RGBShift(p=1, r_shift_limit=(-10, 10), g_shift_limit=(-10, 10), b_shift_limit=(-10, 10)),
            A.ISONoise(p=0.5)
        ]),
        A.Sequential([
            Mosaic(
                dataset,
                transforms=[
                    A.Rotate(limit=5, p=1, border_mode=0),
                    MixUp(dataset, rate_range=(0, 0.1), mix_label=False, p=0.5),
                    A.RandomBrightnessContrast(p=1),
                    A.RGBShift(p=1, r_shift_limit=(-10, 10), g_shift_limit=(-10, 10), b_shift_limit=(-10, 10)),
                    A.ISONoise(p=0.5)
                ],
                always_apply=True
            ),
        ])
    ], p=1),
    
], bbox_params=A.BboxParams(format='albumentations', min_area=0.5, min_visibility=0.2, label_fields=['labels']))

    transformed = Augmentations(dataset, transform)    
    # sample = transformed[0]
    # print(sample['bboxes'], sample['labels'])
    # visualize(sample['image'], sample['bboxes'], sample['labels'], format='albumentations')

    transformed_ds_len = transformed.__len__()
    # print(transformed_ds_len)

    for n in range(AUG_N):
        idxs = list(range(transformed_ds_len))
        random.shuffle(idxs)

        for i in tqdm(range(transformed_ds_len)):
            i = idxs[i]
            number = n * transformed_ds_len + i

            try:
                transformed_data = transformed[i]
                # print(transformed_data['bboxes'], transformed_data['labels'])
                cv2.imwrite(f'{SAVE_DIR}/images/{number}.jpg', transformed_data['image'])
                height, width = transformed_data['image'].shape[:-1]
                write_xml(f"{SAVE_DIR}/annotations", transformed_data['bboxes'], transformed_data['labels'], number, height, width, 'albumentations')

                # visualize(transformed_data['image'], transformed_data['bboxes'], transformed_data['labels'], format='albumentations')

            except:
                pass