import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
import os

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=9000)])
  except RuntimeError as e:
    print(e)


def preprocess_image(image):
    IMG_WIDTH, IMG_HEIGHT = 480, 270
    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.expand_dims(image, axis=0)
    image = tf.keras.applications.efficientnet.preprocess_input(image)

    return image


def load_image(model1, model2, model3, model4, model5, test_set_path, test_df, class_list, mapping):
    result = []
    idx = 0
    for file_name in test_df['id']:
        folder_name = file_name[0]
        original_test_image = cv2.imread(test_set_path + '/' + folder_name + '/' + file_name + '.JPG')
                
        test_image = preprocess_image(original_test_image)
        
        m1_pred = model1.predict(test_image)
        m2_pred = model2.predict(test_image)
        m3_pred = model3.predict(test_image)
        m4_pred = model4.predict(test_image)
        m5_pred = model5.predict(test_image)

        # soft voting
        avg = (m1_pred[0] + m2_pred[0] + m3_pred[0] + m4_pred[0] + m5_pred[0]) / 5
        landmark_id = np.argmax(avg)
        landmark_name = str(class_list[landmark_id])
        conf = avg[landmark_id]

        print(idx, file_name, landmark_name, landmark_id, conf)
        idx += 1
        result.append({'id' : file_name, 'landmark_id' : landmark_id, 'conf' : conf})

    result_df = pd.DataFrame(result)
    result_df.to_csv("/data/backup/pervinco_2020/datasets/data/public/2020.11.23.csv", index=False)


model1 = "/data/backup/pervinco_2020/model/landmark_classification/test/1_landmark_classification.h5"
model2 = "/data/backup/pervinco_2020/model/landmark_classification/test/2_landmark_classification.h5"
model3 = "/data/backup/pervinco_2020/model/landmark_classification/test/3_landmark_classification.h5"
model4 = "/data/backup/pervinco_2020/model/landmark_classification/test/4_landmark_classification.h5"
model5 = "/data/backup/pervinco_2020/model/landmark_classification/test/5_landmark_classification.h5"

model1 = tf.keras.models.load_model(model1)
model2 = tf.keras.models.load_model(model2)
model3 = tf.keras.models.load_model(model3)
model4 = tf.keras.models.load_model(model4)
model5 = tf.keras.models.load_model(model5)

mapping_file_path = '/data/backup/pervinco_2020/datasets/data/public/category.csv'
mapping = pd.read_csv(mapping_file_path)
CLASS_LIST = mapping['landmark_name'].tolist()
# print(CLASS_LIST)
print("NUM OF CLASSES : ", len(CLASS_LIST))

# Load_test_images
test_set_path = '/data/backup/pervinco_2020/datasets/data/public/test'
test_csv = '/data/backup/pervinco_2020/datasets/data/public/sample_submission.csv'
test_df = pd.read_csv(test_csv)
load_image(model1, model2, model3, model4, model5, test_set_path, test_df, CLASS_LIST, mapping)
