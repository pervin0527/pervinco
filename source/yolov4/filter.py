import glob
import os

imgs = os.listdir("/data/backup/pervinco_2020/darknet/build/darknet/x64/data/obj/COCO2017/JPEGImages")
# print(imgs[:3])

txts = os.listdir("/data/backup/pervinco_2020/darknet/build/darknet/x64/data/obj/COCO2017/labels")
# print(txts[:3])

cnt = 0
for img in imgs:
    img_name = img.split('.')[0]

    if (img_name + ".txt") in txts:
        pass

    else:
        cnt+=1
        os.system("sudo mv /data/backup/pervinco_2020/darknet/build/darknet/x64/data/obj/COCO2017/JPEGImages/" + img + " /data/backup/pervinco_2020/darknet/build/darknet/x64/data/obj/test/")

print(cnt)