import pandas as pd
import cv2
import os

datasets = pd.read_csv('/data/backup/pervinco_2020/colab_dataset/dog-breed-identification/labels.csv')
images_path = '/data/backup/pervinco_2020/colab_dataset/dog-breed-identification/train'
output_path = '/data/backup/pervinco_2020/colab_dataset/dog-breed-identification/Distinct'

ids = datasets['id']
breeds = datasets['breed']

for id, breed in zip(ids, breeds):
    print(id, breed)

    image = cv2.imread(images_path + '/' + id + '.jpg')
    # cv2.imshow('image', image)
    # cv2.waitKey(0)

    if not(os.path.isdir(output_path + '/' + breed)):
        os.makedirs(output_path + '/' + breed)

    else:
        pass

    cv2.imwrite(output_path + '/' + breed + '/' + id + '.jpg', image)
    