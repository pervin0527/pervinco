import cv2
import pathlib
import albumentations as A

ds_path = "/data/backup/pervinco_2020/datasets/landmark_classification"
ds_path = pathlib.Path(ds_path)
ds_path = pathlib.Path(ds_path)

images = list(ds_path.glob('*/*'))
images = sorted([str(path) for path in images])
len_images = len(images)
print(len_images)

for img in images:
    print(img)
    image = cv2.imread(img)
    transform = A.Compose([
                    A.CenterCrop(450, 225),
                    A.ShiftScaleRotate(shift_limit=0, rotate_limit=0),
                    A.Resize(456, 456)
                    ])
    augmented_image = transform(image = image)["image"]
    cv2.imshow("aug", augmented_image)
    cv2.imshow("orig", image)

    cv2.waitKey(0)