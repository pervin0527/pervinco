import tensorflow as tf
import cv2
import glob
import csv
import sys
import numpy as np
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input

IMG_RESIZE = 224
data_generator = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=preprocess_input)
MODEL_PATH = sys.argv[1]
DATASET_NAME = MODEL_PATH.split('/')[-3]
print(DATASET_NAME)
img_path = sorted(glob.glob('/home/barcelona/pervinco/datasets/' + DATASET_NAME + '/valid/*/*'))
CLASS_NAMES = sorted(glob.glob('/home/barcelona/pervinco/datasets/' + DATASET_NAME + '/train/*'))
print('num of testset : ', len(img_path))

labels = []
for i in CLASS_NAMES:
    label = i.split('/')[-1]
    labels.append(label)

print(len(labels))


def write_csv(file_info, labels, anw):
    with open('/home/barcelona/pervinco/test_code/cls_result.csv', 'a') as df:
        write = csv.writer(df, delimiter=',')
        write.writerow([file_info, labels, anw])


if __name__ == "__main__":
    model = tf.keras.models.load_model(MODEL_PATH)

    for image in img_path:
        file_info = image.split('/')[-1]
        # using cv2
        # image = cv2.imread(image)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = cv2.resize(image, (IMG_RESIZE, IMG_RESIZE))

        # using PIL
        image = Image.open(image).convert('RGB')
        image = image.resize((IMG_RESIZE, IMG_RESIZE))
        image = np.array(image)

        image = np.expand_dims(image, axis=0)
        # data_generator.fit(image)
        # image = data_generator.flow(image)
        image = preprocess_input(image)
        predictions = model.predict(image, steps=1)
        score = np.argmax(predictions[0])
        
        # print('answer :', answer)
        print("Input file : ", file_info, 'predict result :', labels[score])
        file_info = file_info.split('_')[0]
        # print(file_info)

        if file_info == str(labels[score]):
            anw = 1
            write_csv(file_info, labels[score], anw)
        else:
            anw = 0
            write_csv(file_info, labels[score], anw)

