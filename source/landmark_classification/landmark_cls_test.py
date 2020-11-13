import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
import os

# GPU restrict
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6000)])
  except RuntimeError as e:
    print(e)


def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (480, 270))
    image = np.expand_dims(image, axis=0)
    image = tf.keras.applications.efficientnet.preprocess_input(image)

    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = image / 255.0
    # image = cv2.resize(image, (480, 270))

    return image


def prediction(model, image, class_list, mapping):
    pred = model.predict(image)
    pred_result = np.argmax(pred[0])
    class_name = str(class_list[pred_result])
    pred_score = pred[0][pred_result]

    # landmark_id = mapping[mapping['landmark_name'] == class_name]['landmark_id'].values[0]
 
    # return landmark_id, class_name, pred_score
    return pred_result, class_name, pred_score


def load_image(model, test_set_path, test_df, class_list, mapping):
    result = []
    for file_name in test_df['id']:
        folder_name = file_name[0]
        original_test_image = cv2.imread(test_set_path + '/' + folder_name + '/' + file_name + '.JPG')
                
        test_image = preprocess_image(original_test_image)
        landmark_id, landmark_name, conf = prediction(model, test_image, class_list, mapping)

        print(landmark_id, landmark_name, conf)
        result.append({'id' : file_name, 'landmark_id' : landmark_id, 'conf' : conf})
        # cv2.imshow('test', original_test_image)
        # cv2.waitKey(0)

    result_df = pd.DataFrame(result)
    print(result_df)
    result_df.to_csv("/data/backup/pervinco_2020/datasets/data/public/v100_test.csv", index=False)


# Load model
model_path = '/data/backup/pervinco_2020/model/landmark_classification/2020.11.12_08:38_tf2/landmark_classification.h5'
model = tf.keras.models.load_model(model_path)
model.summary()

mapping_file_path = '/data/backup/pervinco_2020/datasets/data/public/category.csv'
mapping = pd.read_csv(mapping_file_path)
CLASS_LIST = mapping['landmark_name'].tolist()
# print(CLASS_LIST)
print("NUM OF CLASSES : ", len(CLASS_LIST))

# Load_test_images
test_set_path = '/data/backup/pervinco_2020/datasets/data/public/test'
test_csv = '/data/backup/pervinco_2020/datasets/data/public/sample_submission.csv'
test_df = pd.read_csv(test_csv)
load_image(model, test_set_path, test_df, CLASS_LIST, mapping)
