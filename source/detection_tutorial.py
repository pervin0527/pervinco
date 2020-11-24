import os
import cv2
import glob
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

BASE_PATH = "/data/backup/pervinco_2020/multi-class-bbox-regression/dataset"
IMAGES_PATH = os.path.join(BASE_PATH, "images")
ANNOTS_PATH = os.path.join(BASE_PATH, "annotations")
OUTPUT_PATH = "/data/backup/pervinco_2020/model"

INIT_LR = 1e-4
NUM_EPOCHS = 20
BATCH_SIZE = 32
CLASSES = []

data = []
labels = []
bboxes = []
imagePaths = []

csv_files = sorted(glob.glob(ANNOTS_PATH + '/*.csv'))
for csv in csv_files:
    rows = open(csv).read().strip().split("\n")
    
    for row in rows:
        row = row.split(",")
        (filename, startX, startY, endX, endY, label) = row

        imagePath = IMAGES_PATH + '/' + label + '/' + filename
        image = cv2.imread(imagePath)
        # image = image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        (h, w) = image.shape[:2]

        startX = float(startX) / w
        startY = float(startY) / h
        endX = float(endX) / w
        endY = float(endY) / h

        image = cv2.resize(image, (224, 224))
        # image = np.expand_dims(image, axis=0)
        image = tf.keras.applications.efficientnet.preprocess_input(image)
        

        if label not in CLASSES:
            CLASSES.append(label)

        data.append(image)
        labels.append(CLASSES.index(label))
        bboxes.append((startX, startY, endX, endY))
        imagePaths.append(imagePath)

data = np.array(data, dtype="float32")
labels = np.array(labels)
bboxes = np.array(bboxes, dtype="float32")
imagePaths = np.array(imagePaths)
labels = tf.keras.utils.to_categorical(labels, num_classes=len(CLASSES), dtype='float32')
print(len(CLASSES))

split = train_test_split(data, labels, bboxes, imagePaths, test_size=.2, random_state=42)

(trainImages, testImages) = split[:2]
(trainLabels, testLabels) = split[2:4]
(trainBBoxes, testBBoxes) = split[4:6]
(trainPaths, testPaths) = split[6:]

efn = tf.keras.applications.EfficientNetB0(weights="imagenet",
                                           include_top=False,
                                           input_tensor = tf.keras.Input(shape=(224, 224, 3)))
efn.trainable = True
flatten = efn.output
flatten = tf.keras.layers.Flatten()(flatten)

bboxHead = tf.keras.layers.Dense(128, activation="relu")(flatten)
bboxHead = tf.keras.layers.Dense(64, activation="relu")(bboxHead)
bboxHead = tf.keras.layers.Dense(32, activation="relu")(bboxHead)
bboxHead = tf.keras.layers.Dense(4, activation="sigmoid", name="bounding_box")(bboxHead)

softmaxHead = tf.keras.layers.Dense(512, activation="relu")(flatten)
softmaxHead = tf.keras.layers.Dropout(0.5)(softmaxHead)
softmaxHead = tf.keras.layers.Dense(512, activation="relu")(softmaxHead)
softmaxHead = tf.keras.layers.Dropout(0.5)(softmaxHead)
softmaxHead = tf.keras.layers.Dense(len(CLASSES), activation="softmax",	name="class_label")(softmaxHead)

model = tf.keras.Model(inputs=efn.input, outputs=(bboxHead, softmaxHead))

losses = {
	"class_label": "categorical_crossentropy",
	"bounding_box": "mean_squared_error",
}

lossWeights = {
	"class_label": 1.0,
	"bounding_box": 1.0
}

opt = tf.keras.optimizers.Adam(lr=INIT_LR)
model.compile(loss=losses, optimizer=opt, metrics=["accuracy"], loss_weights=lossWeights)
print(model.summary())

trainTargets = {
	"class_label": trainLabels,
	"bounding_box": trainBBoxes
}

testTargets = {
	"class_label": testLabels,
	"bounding_box": testBBoxes
}

H = model.fit(
	trainImages, trainTargets,
	validation_data=(testImages, testTargets),
	batch_size=BATCH_SIZE,
	epochs=NUM_EPOCHS,
	verbose=1)

model.save(OUTPUT_PATH + '/' + 'detection_test3.h5')