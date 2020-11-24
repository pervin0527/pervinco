import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
import os
from collections import Counter

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=9000)])
  except RuntimeError as e:
    print(e)


def preprocess_image(image):
    IMG_WIDTH, IMG_HEIGHT = 300, 300

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = np.float32(image / 255.0)
    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    image = np.expand_dims(image, axis=0)
    image = tf.keras.applications.efficientnet.preprocess_input(image)

    return image


def preprocess_image2(image):
    IMG_WIDTH, IMG_HEIGHT = 480, 270

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.float32(image / 255.0)
    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    image = np.expand_dims(image, axis=0)
    #image = tf.keras.applications.efficientnet.preprocess_input(image)

    return image


def hard_voting(voting_list):
    return Counter(voting_list).most_common(n=1)[0][0]
    


def load_image(n1, n2, n3, n4, n5, n6, test_set_path, test_df, class_list, mapping):
    result = []
    idx = 0
    for file_name in test_df['id']:
        folder_name = file_name[0]
        original_test_image = cv2.imread(test_set_path + '/' + folder_name + '/' + file_name + '.JPG')

        input_image = preprocess_image(original_test_image)        
        input_image2 = preprocess_image2(original_test_image)
        
        n1_pred = n1.predict(input_image2)
        n2_pred = n2.predict(input_image2)
        n3_pred = n3.predict(input_image2)
        n4_pred = n4.predict(input_image2)
        n5_pred = n5.predict(input_image2)
        n6_pred = n6.predict(input_image)

        # soft voting
        avg = (n1_pred[0] + n2_pred[0] + n3_pred[0] + n4_pred[0] + n5_pred[0] + n6_pred[0]) / 6
        landmark_id = np.argmax(avg)
        landmark_name = str(class_list[landmark_id])
        conf = avg[landmark_id]

        # hard voting
        # voting_list = [np.argmax(n1_pred[0]), np.argmax(n2_pred[0]), np.argmax(n3_pred[0]), np.argmax(n4_pred[0]), np.argmax(n5_pred[0])] 
        # landmark_id = hard_voting(voting_list)
        # landmark_name = str(class_list[landmark_id])
        # conf = n5_pred[0][landmark_id]


        print(idx, file_name, landmark_name, landmark_id, conf)
        idx += 1
        result.append({'id' : file_name, 'landmark_id' : landmark_id, 'conf' : conf})

    result_df = pd.DataFrame(result)
    result_df.to_csv("/data/backup/pervinco_2020/datasets/data/public/2020.11.16_kfold_5.csv", index=False)


efn_1 = "/data/backup/pervinco_2020/model/landmark_classification/kfold_train/1_landmark_classification.h5"
efn_2 = "/data/backup/pervinco_2020/model/landmark_classification/kfold_train/2_landmark_classification.h5"
efn_3 = "/data/backup/pervinco_2020/model/landmark_classification/kfold_train/3_landmark_classification.h5"
efn_4 = "/data/backup/pervinco_2020/model/landmark_classification/kfold_train/4_landmark_classification.h5"
efn_5 = "/data/backup/pervinco_2020/model/landmark_classification/kfold_train/5_landmark_classification.h5"
enf_6 = "/data/backup/pervinco_2020/model/landmark_classification/2020.11.04_05:07_tf2/landmark_classification.h5"

n1 = tf.keras.models.load_model(efn_1)
n2 = tf.keras.models.load_model(efn_2)
n3 = tf.keras.models.load_model(efn_3)
n4 = tf.keras.models.load_model(efn_4)
n5 = tf.keras.models.load_model(efn_5)
n6 = tf.keras.models.load_model(enf_6)

mapping_file_path = '/data/backup/pervinco_2020/datasets/data/public/category.csv'
mapping = pd.read_csv(mapping_file_path)
CLASS_LIST = mapping['landmark_name'].tolist()
# print(CLASS_LIST)
print("NUM OF CLASSES : ", len(CLASS_LIST))

# Load_test_images
test_set_path = '/data/backup/pervinco_2020/datasets/data/public/test'
test_csv = '/data/backup/pervinco_2020/datasets/data/public/sample_submission.csv'
test_df = pd.read_csv(test_csv)
load_image(n1, n2, n3, n4, n5, n6, test_set_path, test_df, CLASS_LIST, mapping)
