import numpy as np
import pandas as pd
import os, cv2, random, time, shutil, csv
import tensorflow as tf
import glob
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pathlib
from tqdm import tqdm

from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization, Dense, GlobalAveragePooling2D, Lambda, Dropout, InputLayer, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.callbacks import EarlyStopping


model_path = "/home/v100/tf_workspace/models/landmark_model.h5"
model = tf.keras.models.load_model(model_path)
model.summary()

mapping_file_path = "/home/v100/tf_workspace/source/category.csv"
mapping = pd.read_csv(mapping_file_path)
CLASS_LIST = mapping['landmark_name'].tolist()
class_to_num = dict(zip(CLASS_LIST, range(len(CLASS_LIST))))
print("NUM OF CLASSES : ", len(CLASS_LIST))

test_set_path = "/home/v100/tf_workspace/test_datasets/land_mark_classification/test"
test_csv = "/home/v100/tf_workspace/test_datasets/land_mark_classification/sample_submisstion.csv"
test_df = pd.read_csv(test_csv)

IMG_SIZE = 331


def load_image(test_set_path, test_df, img_size = (IMG_SIZE, IMG_SIZE, 3)):
    test_images = test_df['id']
    test_data_size = len(test_images)
    
    X = np.zeros([test_data_size, img_size[0], img_size[1], 3], dtype=np.uint8)
    for i in tqdm(range(test_data_size)):
        image_name = test_images[i]
        folder = image_name[0]

        img_dir = test_set_path + '/' + folder + '/' + image_name + '.JPG'
        img_pixels = tf.keras.preprocessing.image.load_img(img_dir, target_size=img_size)
        X[i] = img_pixels

    print('Ouptut Data Size: ', X.shape)
    return X


def get_features(model_name, data_preprocessor, input_size, data):
    input_layer = Input(input_size)
    preprocessor = Lambda(data_preprocessor)(input_layer)
    base_model = model_name(weights='imagenet', include_top=False,
                            input_shape=input_size)(preprocessor)
    avg = GlobalAveragePooling2D()(base_model)
    feature_extractor = Model(inputs = input_layer, outputs = avg)
    feature_maps = feature_extractor.predict(data, batch_size=32, verbose=1)
    print('Feature maps shape: ', feature_maps.shape)

    return feature_maps

test_data = load_image(test_set_path, test_df)

from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
inception_preprocessor = preprocess_input
inception_features = get_features(InceptionV3, inception_preprocessor, (IMG_SIZE, IMG_SIZE, 3), test_data)

from tensorflow.keras.applications.xception import Xception, preprocess_input
xception_preprocessor = preprocess_input
xception_features = get_features(Xception, xception_preprocessor, (IMG_SIZE, IMG_SIZE, 3), test_data)

from tensorflow.keras.applications.nasnet import NASNetLarge, preprocess_input
nasnet_preprocessor = preprocess_input
nasnet_features = get_features(NASNetLarge, nasnet_preprocessor, (IMG_SIZE, IMG_SIZE, 3), test_data)

from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
inc_resnet_preprocessor = preprocess_input
inc_resnet_features = get_features(InceptionResNetV2, inc_resnet_preprocessor, (IMG_SIZE, IMG_SIZE, 3), test_data)

test_features = np.concatenate([inception_features,
                                 xception_features,
                                 nasnet_features,
                                 inc_resnet_features],axis=-1)
print('Final feature maps shape', test_features.shape)

y_pred = model.predict(test_features, batch_size=128)

for b in CLASS_LIST:
    test_df[b] = y_pred[:,class_to_num[b]]

test_df.to_csv('/home/v100/tf_workspace/models/landmark_model_result.csv', index=None)