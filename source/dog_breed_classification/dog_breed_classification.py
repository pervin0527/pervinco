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

from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.applications.xception import Xception, preprocess_input
from tensorflow.keras.applications.nasnet import NASNetLarge, preprocess_input
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
        tf.config.experimental.set_virtual_device_configuration(gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10000)])
  except RuntimeError as e:
    # 프로그램 시작시에 메모리 증가가 설정되어야만 합니다
    print(e)

strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
BATCH_SIZE = 4 * strategy.num_replicas_in_sync
IMG_SIZE = 331

def get_num_files(path):
    '''
    폴더 안에 파일 수를 카운팅
    '''
    if not os.path.exists(path):
        return 0
    return sum(len(files) for r, d, files in os.walk(path))

train_dir = './datasets/dog-breed-identification/train'
test_dir = './datasets/dog-breed-identification/test'

data_size = get_num_files(train_dir)
test_size = get_num_files(test_dir)

print("Data samples size : ", data_size)
print("Test samples size : ", test_size)

labels_dataframe = pd.read_csv('./datasets/dog-breed-identification/labels.csv')
sample_df = pd.read_csv('./datasets/dog-breed-identification/sample_submission.csv')

print(labels_dataframe.head(5))

dog_breeds = sorted(list(set(labels_dataframe['breed']))) # label 리스트 생성. set()으로 중복 제거
n_classes = len(dog_breeds)

print(dog_breeds)
print(n_classes)

class_to_num = dict(zip(dog_breeds, range(n_classes)))
print(class_to_num) # {'affenpinscher': 0, 'afghan_hound': 1, 'african_hunting_dog': 2, 'airedale': 3}


def images_to_array(data_dir, labels_dataframe, img_size = (IMG_SIZE, IMG_SIZE, 3)):
    images_names = labels_dataframe['id']
    images_labels = labels_dataframe['breed']
    data_size = len(images_names)

    X = np.zeros([data_size, img_size[0], img_size[1], img_size[2]], dtype = np.uint8)
    print(X.shape) # (10222, 224, 224, 3)
    y = np.zeros([data_size, 1], dtype = np.uint8)
    print(y.shape) # (10222, 1)

    for i in tqdm(range(data_size)):
        image_name = images_names[i]
        img_dir = os.path.join(data_dir, image_name + '.jpg') # data_path + image_filename + .jpg
        img_pixels = load_img(img_dir, target_size = img_size) # tf.keras.preprocessing.image.load_img - Loads an image into PIL format
        X[i] = img_pixels

        image_breed = images_labels[i]
        y[i] = class_to_num[image_breed]

    y = to_categorical(y) # tf.keras.utils.to_categorical - converts a class vector(integers) to binary class metrix
    ind = np.random.permutation(data_size) # array를 복사해서 shuffle. 일반적인 shuffle은 array자체를 shuffle해서 기존 array가 변경되는 형태임.
    X = X[ind]
    y = y[ind]
    print('Output Data size: ', X.shape)
    print('Output Label size : ', y.shape)

    return X, y

X, y = images_to_array(train_dir, labels_dataframe, img_size = (IMG_SIZE, IMG_SIZE, 3))

def get_features(model_name, data_preprocessor, input_size, data):
    input_layer = Input(input_size)
    preprocessor = Lambda(data_preprocessor)(input_layer)
    base_model = model_name(weights='imagenet', include_top=False,
                            input_shape=input_size)(preprocessor)
    avg = GlobalAveragePooling2D()(base_model)
    feature_extractor = Model(inputs = input_layer, outputs = avg)
    feature_maps = feature_extractor.predict(data, batch_size=BATCH_SIZE, verbose=1)
    print('Feature maps shape: ', feature_maps.shape)

    return feature_maps


def images_to_array2(data_dir, labels_dataframe, img_size = (224,224,3)):
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


with strategy.scope():
    inception_preprocessor = preprocess_input
    inception_features = get_features(InceptionV3, inception_preprocessor, (IMG_SIZE, IMG_SIZE, 3), X)

    xception_preprocessor = preprocess_input
    xception_features = get_features(Xception, xception_preprocessor, (IMG_SIZE, IMG_SIZE, 3), X)

    nasnet_preprocessor = preprocess_input
    nasnet_features = get_features(NASNetLarge, nasnet_preprocessor, (IMG_SIZE, IMG_SIZE, 3), X)

    inc_resnet_preprocessor = preprocess_input
    inc_resnet_features = get_features(InceptionResNetV2, inc_resnet_preprocessor, (IMG_SIZE, IMG_SIZE, 3), X)

    final_features = np.concatenate([inception_features,
                                    xception_features,
                                    nasnet_features,
                                    inc_resnet_features,], axis=-1)

    print('Final feature maps shape', final_features.shape)

    dnn = tf.keras.models.Sequential([
        InputLayer(final_features.shape[1:]),
        Dropout(0.7),
        Dense(n_classes, activation='softmax')
    ])

    dnn.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics=['accuracy'])

EarlyStop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                      patience = 5,
                                                      restore_best_weights = True)
my_callback = [EarlyStop_callback]

history = dnn.fit(final_features, y,
                    batch_size = BATCH_SIZE,
                    epochs = 100,
                    validation_split = 0.1,
                    callbacks = my_callback)



test_data = images_to_array2(test_dir, sample_df, (IMG_SIZE, IMG_SIZE, 3))

inception_features = get_features(InceptionV3, inception_preprocessor, (IMG_SIZE, IMG_SIZE, 3), test_data)
xception_features = get_features(Xception, xception_preprocessor, (IMG_SIZE, IMG_SIZE, 3), test_data)
nasnet_features = get_features(NASNetLarge, nasnet_preprocessor, (IMG_SIZE, IMG_SIZE, 3), test_data)
inc_resnet_features = get_features(InceptionResNetV2, inc_resnet_preprocessor, (IMG_SIZE, IMG_SIZE, 3), test_data)

test_features = np.concatenate([inception_features,
                                 xception_features,
                                 nasnet_features,
                                 inc_resnet_features],axis=-1)

print('Final feature maps shape', test_features.shape)


y_pred = dnn.predict(test_features, batch_size=BATCH_SIZE)

for b in dog_breeds:
    sample_df[b] = y_pred[:,class_to_num[b]]

sample_df.to_csv('pred.csv', index=None)