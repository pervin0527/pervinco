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

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        print("True")
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

def images_to_array2(data_dir, labels_dataframe, img_size = (224,224,3)):
    '''
    Do same as images_to_array but omit some unnecessary steps for test data.
    '''
    images_names = labels_dataframe['id']
    data_size = len(images_names)
    X = np.zeros([data_size, img_size[0], img_size[1], 3], dtype=np.uint8)
    
    for i in tqdm(range(data_size)):
        image_name = images_names[i]
        img_dir = os.path.join(data_dir, image_name+'.jpg')
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


##############################################################################################################################
test_dir = "/home/v100/tf_workspace/test_datasets/dog_breeds_classification/test/"
sample_df = pd.read_csv("/home/v100/tf_workspace/test_datasets/dog_breeds_classification/sample_submission.csv")
model_path = "/home/v100/tf_workspace/models/dog_breeds_classification/test/test.h5"

labels_dataframe = pd.read_csv('/home/v100/tf_workspace/csv/dog_category.csv')
classes = sorted(list(labels_dataframe['landmark_name']))
n_classes = len(classes)
class_to_num = dict(zip(classes, range(n_classes)))

print(class_to_num)
print(n_classes)

img_size = (331, 331, 3)
IMG_SIZE = 331
##############################################################################################################################

model = tf.keras.models.load_model(model_path)
model.summary()
test_data = images_to_array2(test_dir, sample_df, img_size)

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

for c in classes:
    sample_df[c] = y_pred[:, class_to_num[c]]
sample_df.to_csv("/home/v100/tf_workspace/test_datasets/dog_breeds_classification/test_result.csv", index=False)