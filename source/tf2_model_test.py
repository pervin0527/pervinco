import tensorflow as tf
import glob
from tensorflow import keras
import numpy as np
import cv2
import os
from efficientnet.tfkeras import preprocess_input
# from tensorflow.keras.applications.resnet50 import preprocess_input

model_path = '/data/backup/pervinco_2020/model/total_split/2020.06.12_16:39_tf2/total_split.h5'

dataset_name = model_path.split('/')[-3]
test_img_path = '/data/backup/pervinco_2020/Auged_datasets/' + dataset_name + '/test/*.jpg'
class_path = '/data/backup/pervinco_2020/datasets/' + dataset_name

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   try:
#     tf.config.experimental.set_virtual_device_configuration(
#         gpus[0],
#         [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=7000)])
#   except RuntimeError as e:
#     print(e)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_memory_growth(gpus[0], True)
  except RuntimeError as e:
    print(e)

model = tf.keras.models.load_model(model_path)
model.summary()

CLASS_NAMES = sorted(os.listdir(class_path))
print(CLASS_NAMES)
print(len(CLASS_NAMES))

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

    