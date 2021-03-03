import glob
import os

imgs = os.listdir("/data/backup/pervinco_2020/darknet/build/darknet/x64/data/obj/cigar/images")
print(len(imgs))

txts = os.listdir("/data/backup/pervinco_2020/darknet/build/darknet/x64/data/obj/cigar/xmls")
print(len(txts))

cnt = 0
for img in imgs:
    img_name = img.split('.')[0]

    if (img_name + ".xml") in txts:
        pass

    else:
        cnt+=1
        os.system("mv /data/backup/pervinco_2020/darknet/build/darknet/x64/data/obj/cigar/images/" + img + " /data/backup/pervinco_2020/darknet/build/darknet/x64/data/obj/cigar/nope/")

print(cnt)