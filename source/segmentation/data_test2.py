import cv2
import numpy as np

VOC_CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "potted plant",
    "sheep",
    "sofa",
    "train",
    "tv/monitor",
]

VOC_COLORMAP = [
    [0, 0, 0],
    [128, 0, 0],
    [0, 128, 0],
    [128, 128, 0],
    [0, 0, 128],
    [128, 0, 128],
    [0, 128, 128],
    [128, 128, 128],
    [64, 0, 0],
    [192, 0, 0],
    [64, 128, 0],
    [192, 128, 0],
    [64, 0, 128],
    [192, 0, 128],
    [64, 128, 128],
    [192, 128, 128],
    [0, 64, 0],
    [128, 64, 0],
    [0, 192, 0],
    [128, 192, 0],
    [0, 64, 128],
]



check = set()
image = "./VOCdevkit/VOC2012/SegmentationClassRaw/2007_000039.png"
# image = "./dset/annot/train/2007_000033.png"
# image = "./dset/annot/train/2007_000039.png"
image = cv2.imread(image)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        pixel = image[i][j]
        if pixel == 255:
            image[i][j] = 0

        check.add(image[i][j])

print(check)
cv2.imshow('result', image)
cv2.waitKey(0)

# image = "./VOCdevkit/VOC2012/SegmentationClass/2007_000033.png"
# image = cv2.imread(image)
# print(image.shape)
# # cv2.imshow('original', image)
# # cv2.waitKey(0)

# for i in range(len(VOC_COLORMAP)):
#     cond = image == VOC_COLORMAP[i][::-1]
#     cond = np.prod(cond, axis=-1)