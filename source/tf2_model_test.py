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
  # 텐서플로가 첫 번째 GPU에 1GB 메모리만 할당하도록 제한
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5000)])
  except RuntimeError as e:
    # 프로그램 시작시에 가상 장치가 설정되어야만 합니다
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
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    

    predictions = model.predict(image, steps=1)
    index = np.argmax(predictions[0])
    name = str(CLASS_NAMES[index])
    score = str(predictions[0][index])

    print(file_name, name, score)

    