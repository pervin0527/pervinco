import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
import os

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=7000)])
  except RuntimeError as e:
    print(e)


def preprocess_image(image, b_ver):
    if b_ver == "b3":
        IMG_WIDTH, IMG_HEIGHT = 300, 300

    elif b_ver == "b5":
        IMG_WIDTH, IMG_HEIGHT = 456, 456

    elif b_ver =="b6":
        IMG_WIDTH, IMG_HEIGHT = 480, 270

    elif b_ver == "b7":
        IMG_WIDTH, IMG_HEIGHT = 600, 600

    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.expand_dims(image, axis=0)
    image = tf.keras.applications.efficientnet.preprocess_input(image)

    return image

def preprocess_image2(image, b_ver):
    IMG_WIDTH, IMG_HEIGHT = 480, 270

    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.expand_dims(image, axis=0)
    #image = tf.keras.applications.efficientnet.preprocess_input(image)
    image = np.float32(image / 255.0)

    return image



def load_image(b6, b3_2, b5, b7, test_set_path, test_df, class_list, mapping):
    result = []
    idx = 0
    for file_name in test_df['id']:
        folder_name = file_name[0]
        original_test_image = cv2.imread(test_set_path + '/' + folder_name + '/' + file_name + '.JPG')
                
        b3_image = preprocess_image(original_test_image, "b3")
        b5_image = preprocess_image(original_test_image, "b5")
        b6_image = preprocess_image2(original_test_image, "b6")
        b7_image = preprocess_image(original_test_image, "b7")
        
        # b3_pred = b3.predict(b3_image)
        b3_2_pred = b3_2.predict(b3_image)
        b5_pred = b5.predict(b5_image)
        b6_pred = b6.predict(b6_image)
        b7_pred = b7.predict(b7_image)

        # soft voting
        avg = (b6_pred[0] + b3_2_pred[0] + b5_pred[0] + b7_pred[0]) / 4
        landmark_id = np.argmax(avg)
        landmark_name = str(class_list[landmark_id])
        conf = avg[landmark_id]

        print(idx, file_name, landmark_name, landmark_id, conf)
        idx += 1
        result.append({'id' : file_name, 'landmark_id' : landmark_id, 'conf' : conf})

    result_df = pd.DataFrame(result)
    result_df.to_csv("/data/backup/pervinco_2020/datasets/data/public/2020.11.14.csv", index=False)


# efn_b3 = "/data/backup/pervinco_2020/model/landmark_classification/2020.10.26_16:39_tf2/landmark_classification.h5"
efn_b3_2 = "/data/backup/pervinco_2020/model/landmark_classification/2020.11.04_05:07_tf2/landmark_classification.h5" # multi-label classification model
efn_b5 = "/data/backup/pervinco_2020/model/landmark_classification/2020.11.02_11:16_tf2/landmark_classification.h5"
efn_b6 = "/data/backup/pervinco_2020/model/landmark_classification/kfold_train/3_landmark_classification.h5"
enf_b7 = "/data/backup/pervinco_2020/model/landmark_classification/B7/landmark_classification.h5"

# b3 = tf.keras.models.load_model(efn_b3)
b3_2 = tf.keras.models.load_model(efn_b3_2)
b5 = tf.keras.models.load_model(efn_b5)
b6 = tf.keras.models.load_model(efn_b6)
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
load_image(b6, b3_2, b5, b7, test_set_path, test_df, CLASS_LIST, mapping)
