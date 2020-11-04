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


AUTOTUNE = tf.data.experimental.AUTOTUNE
strategy = tf.distribute.experimental.CentralStorageStrategy()
BATCH_SIZE = 32
IMG_SIZE = 331
NUM_EPOCHS = 100
EARLY_STOP_PATIENCE = 3

#####################################################################################################################################
train_dir = '/home/v100/tf_workspace/datasets/landmark_classification'

labels_dataframe = pd.read_csv('/home/v100/tf_workspace/csv/category.csv')
classes = sorted(list(labels_dataframe['landmark_name']))
n_classes = len(classes)
class_to_num = dict(zip(classes, range(n_classes)))

print(class_to_num)
print(n_classes)
#####################################################################################################################################

def images_to_array(data_dir, img_size = (IMG_SIZE, IMG_SIZE, 3)):
    ds_path = pathlib.Path(data_dir)

    images = list(ds_path.glob('*/*'))
    images = sorted([str(path) for path in images])
    random.shuffle(images)
    len_images = len(images)

    X = np.zeros([len_images, img_size[0], img_size[1], img_size[2]], dtype = np.uint8)

    labels = sorted(item.name for item in ds_path.glob('*/') if item.is_dir())
    labels_len = len(labels)
    labels = dict((name, index) for index, name in enumerate(labels))
    labels = [labels[pathlib.Path(path).parent.name] for path in images]
    
    idx = 0
    for img, label in zip(images, labels):
        img_pixels = tf.keras.preprocessing.image.load_img(img, target_size=img_size)
        X[idx] = img_pixels
        idx += 1
        print(idx, img.split('/')[-2], label)

    y = tf.keras.utils.to_categorical(labels, num_classes=labels_len, dtype='float32')

    # ind = np.random.permutation(len_images)
    # X = X[ind]
    # y = y[ind]
    print('Output Data size: ', X.shape)
    print('Output Label size : ', y.shape)

    return X, y


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


def build_lrfn(lr_start=0.00001, lr_max=0.00005, 
               lr_min=0.00001, lr_rampup_epochs=5, 
               lr_sustain_epochs=0, lr_exp_decay=.8):
    lr_max = lr_max * strategy.num_replicas_in_sync

    def lrfn(epoch):
        if epoch < lr_rampup_epochs:
            lr = (lr_max - lr_start) / lr_rampup_epochs * epoch + lr_start
        elif epoch < lr_rampup_epochs + lr_sustain_epochs:
            lr = lr_max
        else:
            lr = (lr_max - lr_min) *\
                 lr_exp_decay**(epoch - lr_rampup_epochs\
                                - lr_sustain_epochs) + lr_min
        return lr
    return lrfn

X, y = images_to_array(train_dir, img_size = (IMG_SIZE, IMG_SIZE, 3))

from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
inception_preprocessor = preprocess_input
inception_features = get_features(InceptionV3, inception_preprocessor, (IMG_SIZE, IMG_SIZE, 3), X)

from tensorflow.keras.applications.xception import Xception, preprocess_input
xception_preprocessor = preprocess_input
xception_features = get_features(Xception, xception_preprocessor, (IMG_SIZE, IMG_SIZE, 3), X)

from tensorflow.keras.applications.nasnet import NASNetLarge, preprocess_input
nasnet_preprocessor = preprocess_input
nasnet_features = get_features(NASNetLarge, nasnet_preprocessor, (IMG_SIZE, IMG_SIZE, 3), X)

from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
inc_resnet_preprocessor = preprocess_input
inc_resnet_features = get_features(InceptionResNetV2, inc_resnet_preprocessor, (IMG_SIZE, IMG_SIZE, 3), X)

with strategy.scope():
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
    dnn.summary()

cb_early_stopper = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
lrfn = build_lrfn()
lr_schedule = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=1) 

history = dnn.fit(final_features, y,
                    batch_size = BATCH_SIZE,
                    epochs = 1000,
                    validation_split = 0.2,
                    callbacks = [cb_early_stopper, lr_schedule]
)

dnn.save('/home/v100/tf_workspace/models/landmark_model.h5')