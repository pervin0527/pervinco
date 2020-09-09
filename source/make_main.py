import glob

path = sorted(glob.glob('./VOCdevkit/VOC2007/JPEGImages/*'))
print(len(path))

f = open('./VOCdevkit/VOC2007/ImageSets/Main/product_train.txt', 'w')

for image in path:
    file_name = image.split('/')[-1]
    file_name = file_name.split('.')[0]

    print(file_name)

    f.write(file_name + ' -1' + '\n')

f.close()

