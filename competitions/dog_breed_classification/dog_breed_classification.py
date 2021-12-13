import numpy as np
import pandas as pd
import os, cv2, random, time, shutil, csv
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tqdm import tqdm

from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization, Dense, GlobalAveragePooling2D, Lambda, Dropout, InputLayer, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.callbacks import EarlyStopping


def get_num_files(path):
    if not os.path.exists(path):
        return 0
    return sum(len(files) for r, d, files in os.walk(path))


def images_to_array(data_dir, labels_dataframe, img_size = (224,224,3)):
    images_names = labels_dataframe['id']
    images_labels = labels_dataframe['breed']

    data_size = len(images_names)
    X = np.zeros([data_size, img_size[0], img_size[1], img_size[2]], dtype=np.uint8)
    y = np.zeros([data_size,1], dtype=np.uint8)

    for i in tqdm(range(data_size)):
        image_name = images_names[i]
        img_dir = os.path.join(data_dir, image_name+'.jpg')
        # img_pixels = load_img(img_dir, target_size=img_size)
        img_pixels = cv2.imread(img_dir)
        img_pixels = cv2.cvtColor(img_pixels, cv2.COLOR_BGR2RGB)
        img_pixels = cv2.resize(img_pixels, (img_size[0], img_size[1]))
        X[i] = img_pixels
        
        image_breed = images_labels[i]
        y[i] = class_to_num[image_breed]
    
    y = to_categorical(y)
    ind = np.random.permutation(data_size)
    X = X[ind]
    y = y[ind]
    print('Ouptut Data Size: ', X.shape)
    print('Ouptut Label Size: ', y.shape)
    return X, y


def get_features(model_name, data_preprocessor, input_size, data):
    input_layer = Input(input_size)
    preprocessor = Lambda(data_preprocessor)(input_layer)
    base_model = model_name(weights='imagenet', include_top=False,
                            input_shape=input_size)(preprocessor)
    avg = GlobalAveragePooling2D()(base_model)
    feature_extractor = Model(inputs = input_layer, outputs = avg)
    
    feature_maps = feature_extractor.predict(data, batch_size=64, verbose=1)
    print('Feature maps shape: ', feature_maps.shape)
    return feature_maps


train_dir = '/data/tf_workspace/datasets/dog_breeds_classification/train'
test_dir = '/data/tf_workspace/datasets/dog_breeds_classification/test'

data_size = get_num_files(train_dir)
test_size = get_num_files(test_dir)

print("Data samples size : ", data_size)
print("Test samples size : ", test_size)

labels_dataframe = pd.read_csv('/data/tf_workspace/datasets/dog_breeds_classification/labels.csv')
sample_df = pd.read_csv('/data/tf_workspace/datasets/dog_breeds_classification/sample_submission.csv')

print(labels_dataframe.head(5))

dog_breeds = sorted(list(set(labels_dataframe['breed']))) # label 리스트 생성. set()으로 중복 제거
n_classes = len(dog_breeds)
class_to_num = dict(zip(dog_breeds, range(n_classes)))
print(class_to_num) # {'affenpinscher': 0, 'afghan_hound': 1, 'african_hunting_dog': 2, 'airedale': 3}

img_size = (331, 331, 3)
X, y = images_to_array(train_dir, labels_dataframe, img_size = img_size)

from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
inception_preprocessor = preprocess_input
inception_features = get_features(InceptionV3,
                                  inception_preprocessor,
                                  img_size, X)

from tensorflow.keras.applications.xception import Xception, preprocess_input
xception_preprocessor = preprocess_input
xception_features = get_features(Xception,
                                 xception_preprocessor,
                                 img_size, X)

from tensorflow.keras.applications.efficientnet import EfficientNetB2, preprocess_input
effnet_preprocessor = preprocess_input
effnet_features = get_features(EfficientNetB2,
                               effnet_preprocessor,
                               img_size, X)

from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
inc_resnet_preprocessor = preprocess_input
inc_resnet_features = get_features(InceptionResNetV2,
                                   inc_resnet_preprocessor,
                                   img_size, X)

del X

final_features = np.concatenate([inception_features,
                                 xception_features,
                                 effnet_features,
                                 inc_resnet_features,], axis=-1)
print('Final feature maps shape', final_features.shape)

EarlyStop_callback = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
my_callback=[EarlyStop_callback]

dnn = tf.keras.models.Sequential([
    InputLayer(final_features.shape[1:]),
    Dropout(0.7),
    Dense(n_classes, activation='softmax')
])

dnn.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

h = dnn.fit(final_features, y,
            batch_size=128,
            epochs=60,
            validation_split=0.1,
            callbacks=my_callback)

dnn.save("/data/tf_workspace/models/dog_breed_classification/ensemble_model.h5")

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
        img_pixels = cv2.imread(img_dir)
        img_pixels = cv2.cvtColor(img_pixels, cv2.COLOR_BGR2RGB)
        img_pixels = cv2.resize(img_pixels, (img_size[0], img_size[1]))
        X[i] = img_pixels
        
    print('Ouptut Data Size: ', X.shape)
    return X

test_data = images_to_array2(test_dir, sample_df, img_size)

inception_features = get_features(InceptionV3, inception_preprocessor, img_size, test_data)
xception_features = get_features(Xception, xception_preprocessor, img_size, test_data)
effnet_features = get_features(EfficientNetB2, effnet_preprocessor, img_size, test_data)
inc_resnet_features = get_features(InceptionResNetV2, inc_resnet_preprocessor, img_size, test_data)

test_features = np.concatenate([inception_features,
                                 xception_features,
                                 effnet_features,
                                 inc_resnet_features],axis=-1)
print('Final feature maps shape', test_features.shape)

del test_data

y_pred = dnn.predict(test_features, batch_size=128)

print(y_pred)