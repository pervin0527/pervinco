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


def b3_preprocess_image(image):
    image = cv2.resize(image, (300, 300))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.expand_dims(image, axis=0)
    image = tf.keras.applications.efficientnet.preprocess_input(image)

    return image

def b5_preprocess_image(image):
    image = cv2.resize(image, (456, 456))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.expand_dims(image, axis=0)
    image = tf.keras.applications.efficientnet.preprocess_input(image)

    return image

def b7_preprocess_image(image):
    image = cv2.resize(image, (600, 600))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.expand_dims(image, axis=0)
    image = tf.keras.applications.efficientnet.preprocess_input(image)

    return image


def prediction(model, image, class_list, mapping):
    pred = model.predict(image)

    return pred


def load_image(b3, b3_2, b5_2, b7, test_set_path, test_df, class_list, mapping):
    result = []
    idx = 0
    for file_name in test_df['id']:
        folder_name = file_name[0]
        original_test_image = cv2.imread(test_set_path + '/' + folder_name + '/' + file_name + '.JPG')
                
        b3_image = b3_preprocess_image(original_test_image)
        b5_image = b5_preprocess_image(original_test_image)
        b7_image = b7_preprocess_image(original_test_image)
        
        b3_pred = b3.predict(b3_image)
        b3_2_pred = b3_2.predict(b3_image)
        # b5_pred = b5.predict(b5_image)
        b5_2_pred = b5_2.predict(b5_image)
        b7_pred = b7.predict(b7_image)

        # soft voting
        avg = (b3_pred[0] + b3_2_pred[0] + b5_2_pred[0] + b7_pred[0]) / 4
        landmark_id = np.argmax(avg)
        landmark_name = str(class_list[landmark_id])
        conf = avg[landmark_id]

        # weight voting
        # if conf < 0.9:
        #     weight_avg = (b3_pred[0] + b3_2_pred[0] + b5_2_pred[0] + (2 * b7_pred[0])) / 5
        #     landmark_id = np.argmax(weight_avg)
        #     landmark_name = str(class_list[landmark_id])
        #     conf = weight_avg[landmark_id]

        print(idx, file_name, landmark_name, landmark_id, conf)
        idx += 1
        result.append({'id' : file_name, 'landmark_id' : landmark_id, 'conf' : conf})

    result_df = pd.DataFrame(result)
    result_df.to_csv("/data/backup/pervinco_2020/datasets/data/public/final.csv", index=False)


efn_b3 = "/data/backup/pervinco_2020/model/landmark_classification/2020.10.26_16:39_tf2/landmark_classification.h5"
efn_b3_2 = "/data/backup/pervinco_2020/model/landmark_classification/2020.11.04_05:07_tf2/landmark_classification.h5"
# efn_b5 = "/data/backup/pervinco_2020/model/landmark_classification/2020.10.26_10:18_tf2/landmark_classification.h5"
efn_b5_2 = "/data/backup/pervinco_2020/model/landmark_classification/2020.11.02_11:16_tf2/landmark_classification.h5"
enf_b7 = "/data/backup/pervinco_2020/model/landmark_classification/B7/landmark_classification.h5"
b3 = tf.keras.models.load_model(efn_b3)
b3_2 = tf.keras.models.load_model(efn_b3_2)
# b5 = tf.keras.models.load_model(efn_b5) 
b5_2 = tf.keras.models.load_model(efn_b5_2)
b7 = tf.keras.models.load_model(enf_b7)

mapping_file_path = '/data/backup/pervinco_2020/datasets/data/public/category.csv'
mapping = pd.read_csv(mapping_file_path)
CLASS_LIST = mapping['landmark_name'].tolist()
# print(CLASS_LIST)
print("NUM OF CLASSES : ", len(CLASS_LIST))

# Load_test_images
test_set_path = '/data/backup/pervinco_2020/datasets/data/public/test'
test_csv = '/data/backup/pervinco_2020/datasets/data/public/sample_submission.csv'
test_df = pd.read_csv(test_csv)
load_image(b3, b3_2, b5_2, b7, test_set_path, test_df, CLASS_LIST, mapping)
