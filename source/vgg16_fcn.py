import tensorflow as tf
import glob
import os
import cv2
import xml.etree.ElementTree as ET
from xml.dom import minidom
import numpy as np
from sklearn.model_selection import train_test_split

IMG_SIZE = 224
INIT_LR = 1e-4
NUM_EPOCHS = 20
BATCH_SIZE = 32


def class_text_to_int(row_label):
    if row_label == 'aeroplane':
        return 0

    elif row_label == 'bicycle':
        return 1

    elif row_label == 'bird':
        return 2

    elif row_label == 'boat':
        return 3

    elif row_label == 'bottle':
        return 4

    elif row_label == 'bus':
        return 5

    elif row_label == 'car':
        return 6

    elif row_label == 'cat':
        return 7

    elif row_label == 'chair':
        return 8

    elif row_label == 'cow':
        return 9

    elif row_label == 'diningtable':
        return 10

    elif row_label == 'dog':
        return 11

    elif row_label == 'horse':
        return 12

    elif row_label == 'motorbike':
        return 13

    elif row_label == 'person':
        return 14

    elif row_label == 'pottedplant':
        return 15

    elif row_label == 'sheep':
        return 16

    elif row_label == 'sofa':
        return 17

    elif row_label == 'train':
        return 18

    elif row_label == 'tvmonitor':
        return 19

    else:
        pass


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

    o = tf.keras.layers.Conv2D(4096 ,(7, 7), activation='relu' , padding='same', name="conv6")(base_model.output)
    conv7 = tf.keras.layers.Conv2D(4096, (1, 1) , activation='relu' , padding='same', name="conv7")(o)
    conv7_4 = tf.keras.layers.Conv2DTranspose(20, kernel_size=(4,4), strides=(4,4), use_bias=False)(conv7)

    pool411 = tf.keras.layers.Conv2D(20, (1, 1) , activation='relu' , padding='same', name="pool4_11")(base_model.get_layer("block4_pool").output)
    pool411_2 = tf.keras.layers.Conv2DTranspose(20, kernel_size=(2,2), strides=(2,2), use_bias=False)(pool411)

    pool311 = tf.keras.layers.Conv2D(20, (1,1) , activation='relu' , padding='same', name="pool3_11")(base_model.get_layer("block3_pool").output)

    o = tf.keras.layers.Add(name="add")([pool411_2, pool311, conv7_4 ])
    o = tf.keras.layers.Conv2DTranspose(20 , kernel_size=(8,8) ,  strides=(8,8) , use_bias=False)(o)
    o = (tf.keras.layers.Activation('softmax'))(o)
    
    model = tf.keras.Model(base_model.input, o)

    return model


if __name__ == "__main__":
    images = "/data/backup/pervinco_2020/datasets/VOC2012/sample_image"
    images = sorted(glob.glob(images + '/*.jpg'))

    annotations = "/data/backup/pervinco_2020/datasets/VOC2012/sample_annotation"
    annotations = sorted(glob.glob(annotations + '/*.xml'))

    total_images = []
    total_labels = []
    total_boxes = []

    for image, annotation in zip(images, annotations):    
        bboxes, classes = get_boxes(annotation)

        file_name = image.split('/')[-1]
        image = cv2.imread(image)
        h, w = image.shape[:2]
        image = cv2.resize(image, (224, 224))
        image = tf.keras.applications.efficientnet.preprocess_input(image)

        for (xmin, ymin, xmax, ymax), label in zip(bboxes, classes):
            xmin = float(xmin) / w
            ymin = float(ymin) / h
            xmax = float(xmax) / w
            ymax = float(ymax) / h

            total_boxes.append((xmin, ymin, xmax, ymax))
            # print(file_name, class_text_to_int(label))
            total_labels.append(class_text_to_int(label))
            total_images.append(image)

    total_images = np.array(total_images, dtype="float32")
    total_labels = np.array(total_labels)
    total_labels = tf.keras.utils.to_categorical(total_labels, num_classes=20, dtype="float32")
    total_boxes = np.array(total_boxes, dtype="float32")

    split = train_test_split(total_images, total_labels, total_boxes, test_size=.2, random_state=42)

    (trainImages, testImages) = split[:2]
    (trainLabels, testLabels) = split[2:4]
    (trainBBoxes, testBBoxes) = split[4:6]

    model = create_model()
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.summary()
    tf.keras.utils.plot_model(model, to_file="/data/backup/pervinco_2020/test_code/test_model_plot.png")

    H = model.fit(
        trainImages, trainLabels,
        validation_data=(testImages, testLabels),
        batch_size=BATCH_SIZE,
        epochs=NUM_EPOCHS,
        verbose=1)

    model.save('/data/backup/pervinco_2020/model/voc_detection_model.h5')