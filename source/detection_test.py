import tensorflow as tf
import cv2
import numpy as np
import glob

CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
           'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
           'dog', 'horse', 'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor']

model_path ="/data/backup/pervinco_2020/model/voc_detection_model.h5"
model = tf.keras.models.load_model(model_path)

test_image_path = "/data/backup/pervinco_2020/datasets/detection_test"
test_images = sorted(glob.glob(test_image_path + '/*.jpg'))
print(len(test_images))

for test_image in test_images:
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(224, 224))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.keras.applications.efficientnet.preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    
    (boxPreds, labelPreds) = model.predict(image)
    # print(boxPreds, labelPreds)
    (startX, startY, endX, endY) = boxPreds[0]
    
    i = np.argmax(labelPreds, axis=1)
    # print(i)
    label = CLASSES[i[0]]

    image = cv2.imread(test_image)
    image = cv2.resize(image, (640, 480))
    h, w = image.shape[:2]

    startX = int(startX * w)
    startY = int(startY * h)
    endX = int(endX * w)
    endY = int(endY * h)

    y = startY - 10 if startY - 10 > 10 else startY + 10
    cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

    cv2.imshow("Output", image)
    cv2.waitKey(0)