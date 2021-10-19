# https://imgaug.readthedocs.io/en/latest/source/examples_bounding_boxes.html
# https://junyoung-jamong.github.io/machine/learning/2019/01/23/%EB%B0%94%EC%9A%B4%EB%94%A9%EB%B0%95%EC%8A%A4%EB%A5%BC-%ED%8F%AC%ED%95%A8%ED%95%9C-%EC%9D%B4%EB%AF%B8%EC%A7%80-%EC%A6%9D%ED%8F%AD%EC%8B%9C%ED%82%A4%EA%B8%B0-with-imgaug.html

import cv2
import imgaug as ia
import imgaug.augmenters as iaa
import xml.etree.ElementTree as ET
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from xml.dom import minidom

def get_boxes(label_path):
    tree = ET.parse(label_path)
    root = tree.getroot()
    obj_xml = root.findall('object')
    
    if obj_xml[0].find('bndbox') != None:

        result = []
        for obj in obj_xml:
            bbox_original = obj.find('bndbox')
            names = obj.find('name')
        
            xmin = int(bbox_original.find('xmin').text)
            ymin = int(bbox_original.find('ymin').text)
            xmax = int(bbox_original.find('xmax').text)
            ymax = int(bbox_original.find('ymax').text)

            result.append([xmin, ymin, xmax, ymax])
        
        return result


image_path = "./45.jpg"
image = cv2.imread(image_path)
annotation_path = "./45.xml"
annotation_data = get_boxes(annotation_path)
print(annotation_data)

bboxes = []
for box in annotation_data:
    bboxes.append(ia.BoundingBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3]))
bbs = BoundingBoxesOnImage(bboxes, shape=image.shape)

seq = iaa.Sequential([
    iaa.Multiply((1.2, 1.5)), # change brightness, doesn't affect BBs
    iaa.Affine(
        translate_px={"x": 40, "y": 60},
        # scale=(0.5, 0.7),
        rotate=45
    ) # translate by 40/60px on x/y axis, and scale to 50-70%, affects BBs
])

image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)

for i in range(len(bbs.bounding_boxes)):
    before = bbs.bounding_boxes[i]
    after = bbs_aug.bounding_boxes[i]
    print("BB %d: (%.4f, %.4f, %.4f, %.4f) -> (%.4f, %.4f, %.4f, %.4f)" % (
        i,
        before.x1, before.y1, before.x2, before.y2,
        after.x1, after.y1, after.x2, after.y2)
    )
    print()
    print(after)

# image with BBs before/after augmentation (shown below)
image_before = bbs.draw_on_image(image, size=2)
image_after = bbs_aug.draw_on_image(image_aug, size=2, color=[0, 0, 255])

cv2.imshow('result', image_after)
cv2.waitKey(0)