import os
import cv2
from glob import glob

root = "/home/barcelona/바탕화면/naver_map"
rejected = f"{root}/Rejected"
folder = "Paris_baguette"

latest_number = 0
latest = sorted(glob(f"{root}/{folder}/{folder}*"))
if latest:
    latest_number = int(latest[-1].split('/')[-1].split('_')[-1].split('.')[0])

images = sorted(glob(f"{root}/{folder}/스크린샷,*"))
print(len(images))

for number, image_path in enumerate(images):
    number += (latest_number + 1)
    image = cv2.imread(image_path)
    image = cv2.resize(image, (384, 384))

    cv2.imshow(f"{number}", image)
    k = cv2.waitKey(0)

    if k == ord('d'):
        # os.remove(image_path)
        file_name = image_path.split('/')[-1]
        os.rename(image_path, f"{rejected}/{file_name}")

    elif k == 27:
        break

    elif k == ord('p'):
        file_path = ('/').join(image_path.split('/')[:-1])
        os.rename(image_path, f"{file_path}/{folder}_{number}.png")
        pass

    cv2.destroyAllWindows()