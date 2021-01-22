import tensorflow as tf
import glob
import os
import cv2
import xml.etree.ElementTree as ET
from xml.dom import minidom
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

IMG_SIZE = 224
INIT_LR = 1e-4
NUM_EPOCHS = 20
BATCH_SIZE = 32

n_classes = 3
epoch_size = 300
# input_height , input_width = 224 , 672
# output_height , output_width = 224 , 672

dir_data = "/data/backup/pervinco_2020/datasets/segmentation_test/training"
dir_seg = dir_data + "/gt_image/"
dir_img = dir_data + "/image/"


def get_boxes(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    obj_xml = root.findall('object')
    
    if obj_xml[0].find('bndbox') != None:

        bboxes = []
        classes = []

        for obj in obj_xml:
            bbox_original = obj.find('bndbox')
            names = obj.find('name')
        
            xmin = int(float(bbox_original.find('xmin').text))
            ymin = int(float(bbox_original.find('ymin').text))
            xmax = int(float(bbox_original.find('xmax').text))
            ymax = int(float(bbox_original.find('ymax').text))

            bboxes.append((xmin, ymin, xmax, ymax))
            classes.append(names.text)
        
        return bboxes, classes


def create_model():
    base_model = tf.keras.applications.VGG16(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                                                      weights="imagenet",
                                                      include_top=False)

    conv7_7 = tf.keras.layers.Conv2D(4096 ,(7, 7), activation='relu' , padding='same', name="conv6")(base_model.output)
    conv1_1 = tf.keras.layers.Conv2D(4096, (1, 1) , activation='relu' , padding='same', name="conv7")(conv7_7)
    conv7_1_out = tf.keras.layers.Conv2DTranspose(n_classes, kernel_size=(4,4), strides=(4,4), use_bias=False)(conv1_1)

    pool4_1_1 = tf.keras.layers.Conv2D(n_classes, (1, 1) , activation='relu' , padding='same', name="pool1_1")(base_model.get_layer("block4_pool").output)
    pool4_1_1_out = tf.keras.layers.Conv2DTranspose(n_classes, kernel_size=(2,2), strides=(2,2), use_bias=False)(pool4_1_1)

    pool_3_1_1_out = tf.keras.layers.Conv2D(n_classes, (1,1) , activation='relu' , padding='same', name="pool3_1_1")(base_model.get_layer("block3_pool").output)

    out = tf.keras.layers.Add(name="add")([pool4_1_1_out, pool_3_1_1_out, conv7_1_out])
    out = tf.keras.layers.Conv2DTranspose(n_classes , kernel_size=(8,8) ,  strides=(8,8) , use_bias=False)(out)
    out = (tf.keras.layers.Activation('softmax'))(out)
    
    model = tf.keras.Model(base_model.input, out)

    return model


def getImageArr( path , width , height ):
    img = cv2.imread(path, 1)
    img = np.float32(cv2.resize(img, ( width , height ))) / 127.5 - 1
    return img

def getOrigin( path , width , height ):
    img = cv2.imread(path, 1)
    img = np.float32(cv2.resize(img, ( width , height )))
    return img


def getSegmentationArr( path , nClasses ,  width , height  ):

    seg_labels = np.zeros((  height , width  , nClasses ))
    img = cv2.imread(path, 1)
    img = cv2.resize(img, ( width , height ))
    img_normal = cv2.normalize(img, None, 0,2, cv2.NORM_MINMAX)
    img_normal = img_normal[:, : , 0]

    for c in range(nClasses):
        seg_labels[: , : , c ] = (img_normal == c ).astype(int)
    ##seg_labels = np.reshape(seg_labels, ( width*height,nClasses  ))
    return seg_labels


def train(n_classes):
    images = os.listdir(dir_img)
    images.sort()
    segmentations  = os.listdir(dir_seg)
    segmentations.sort()
        
    X = []
    Y = []
    for im , seg in zip(images,segmentations) :
        X.append(getImageArr(dir_img + im , IMG_SIZE , IMG_SIZE))
        Y.append(getSegmentationArr( dir_seg + seg , n_classes , IMG_SIZE , IMG_SIZE))

    X, Y = np.array(X) , np.array(Y)
    Full_dataset = 'Full  dataset : {}, {}'.format(X.shape, Y.shape)
    print(Full_dataset)

    train_rate = 0.85
    index_train = np.random.choice(X.shape[0],int(X.shape[0]*train_rate),replace=False)
    index_valid = list(set(range(X.shape[0])) - set(index_train))
                                
    X, Y = shuffle(X,Y)
    X_train, y_train = X[index_train],Y[index_train]
    X_valid, y_valid = X[index_valid],Y[index_valid]
    Train_dataset = 'Train dataset : {}, {}'.format(X_train.shape, y_train.shape)
    print(Train_dataset)
    Valid_dataset = 'Valid dataset : {}, {}'.format(X_valid.shape, y_valid.shape)
    print(Valid_dataset)

    model = create_model()
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.summary()
    tf.keras.utils.plot_model(model, to_file="/data/backup/pervinco_2020/test_code/test_model_plot.png")

    model.fit(X_train, y_train, validation_data=(X_valid, y_valid), batch_size=32, epochs=epoch_size, verbose=1)

    model.save('/data/backup/pervinco_2020/model/FCN_model.h5')

train(n_classes)