import tensorflow as tf
import glob
from tensorflow import keras
import numpy as np
import cv2
import os

model_path = '/data/backup/pervinco_2020/model/cat_dog_mask/2020.11.13_12:07_tf2/5_cat_dog_mask.h5'

dataset_name = model_path.split('/')[-3]
test_img_path = '/data/backup/pervinco_2020/Auged_datasets/' + dataset_name + '/test/*.jpg'
class_path = '/data/backup/pervinco_2020/datasets/' + dataset_name

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6000)])
  except RuntimeError as e:
    print(e)


model = tf.keras.models.load_model(model_path)
model.summary()

CLASS_NAMES = sorted(os.listdir(class_path))
print(CLASS_NAMES)
print("Num of Classes", len(CLASS_NAMES))

test_imgs = sorted(glob.glob(test_img_path))
print("Num of Test Image : ", len(test_imgs))

for img in test_imgs:
    file_name = img.split('/')[-1]
    original_image = cv2.imread(img)
    image = cv2.resize(original_image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    # image = tf.keras.applications.efficientnet.preprocess_input(image)
    

    predictions = model.predict(image, steps=1)
    index = np.argmax(predictions[0])
    name = str(CLASS_NAMES[index])
    score = str(int(predictions[0][index] * 100)) + '%'

    print(file_name, name, score)

    result_image = cv2.putText(original_image, name + ' : ' + str(score), org=(50,50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=2)
    result_image = cv2.resize(result_image, (512, 512))
    cv2.imshow('result', result_image)
    cv2.waitKey(0)
    # cv2.imwrite('/data/backup/pervinco_2020/Auged_datasets/mask_classification/test/result/result_' + file_name, result_image)

    