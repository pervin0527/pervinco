import cv2
from glob import glob

files = sorted(glob('/home/barcelona/사진/*.png'))
for file in files:
    image = cv2.imread(file)
    resized_image = cv2.resize(image, (384, 384))

    cv2.imshow("resized", resized_image)
    key = cv2.waitKeyEx()

    if key == ord('p'):
        pass

    elif key == 27:
        break