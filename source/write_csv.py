# -*- coding: utf-8 -*-

import csv
import glob
import tensorflow as tf
import numpy as np
from PIL import Image
import math
from tensorflow.keras.applications.resnet50 import preprocess_input

ds_dir = sorted(glob.glob('/home/barcelona/SSD/testset_merge/*'))
model_dir = '/home/barcelona/pervinco/model/cu50/2020.02.20_14:16_keras/cu50.h5'

with open('/home/barcelona/pervinco/cu50_mapping.csv', 'r') as df:
    reader = csv.reader(df)
    CLASS_NAMES = list(reader)


def write_csv(file_name, barcode1, score1, barcode2, score2, barcode3, score3):
    with open('/home/barcelona/SSD/testset_cls_topN.csv', 'a') as df:
        write = csv.writer(df, delimiter=',')
        write.writerow([file_name, barcode1, score1, barcode2, score2, barcode3, score3])


if __name__ == "__main__":
    model = tf.keras.models.load_model(model_dir)

    for image in ds_dir:
        file_name = image.split('/')[-1]

        # open image
        image = Image.open(image).convert('RGB')
        image = image.resize((224, 224))
        image = np.array(image)

        # preprocessing
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)

        # predict
        predictions = model.predict(image, steps=1)
        index = np.argmax(predictions[0])
        # barcode = str(CLASS_NAMES[index][1])
        # score = predictions[0][index]
        # print(file_name, score)
        # write_csv(file_name, barcode, score)

        p = predictions[0]
        p = np.argpartition(p, -3)[-3:]
        # print(p)

        s1 = predictions[0][p[0]]
        s2 = predictions[0][p[1]]
        s3 = predictions[0][p[2]]

        b1 = str(CLASS_NAMES[p[0]][1])
        b2 = str(CLASS_NAMES[p[1]][1])
        b3 = str(CLASS_NAMES[p[2]][1])

        # print(b1, b2, b3)
        print(file_name)
        write_csv(file_name, b1, "{:f}".format(float(s1)), b2, "{:f}".format(float(s2)), b3, "{:f}".format(float(s3)))



