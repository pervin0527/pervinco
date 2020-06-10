import tensorflow as tf
import glob
from tensorflow import keras
import numpy as np
import cv2
import os
from efficientnet.tfkeras import EfficientNetB0, preprocess_input
# from tensorflow.keras.applications.resnet50 import preprocess_input


dataset_name = 'beverage'
class_path = '/data/backup/pervinco_2020/datasets/' + dataset_name + '/train'
model_path = '/data/backup/pervinco_2020/model/beverage/2020.06.09_11:42_tf2/efn_beverage.h5'
test_img_path = '/data/backup/pervinco_2020/datasets/' + dataset_name + '/test/*.jpg'

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        print("True")
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

model = tf.keras.models.load_model(model_path)

CLASS_NAMES = sorted(os.listdir(class_path))
print(CLASS_NAMES)

test_imgs = sorted(glob.glob(test_img_path))
print(len(test_imgs))

for img in test_imgs:
    file_name = img.split('/')[-1]
    image = cv2.imread(img)
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    # image = preprocess_input(image)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)

    predictions = model.predict(image, steps=1)
    index = np.argmax(predictions[0])
    name = str(CLASS_NAMES[index])
    score = str(predictions[0][index])

    print(file_name, name, score)

    