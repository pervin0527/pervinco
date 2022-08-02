import os
import cv2
import numpy as np
import tensorflow as tf

from model import centernet

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 1:
    try:
        print("Activate Multi GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
    except RuntimeError as e:
        print(e)

else:
    try:
        print("Activate Sigle GPU")
        tf.config.experimental.set_memory_growth(gpus[0], True)
        strategy = tf.distribute.experimental.CentralStorageStrategy()
    except RuntimeError as e:
        print(e)

if __name__ == "__main__":
    input_shape = (512, 512, 3)
    backbone = "resnet101"
    classes = ["face"]
    max_detections = 10

    image_path = "./samples/sample01.jpg"
    ckpt_path = "/home/ubuntu/Models/centernet.h5"
    model, prediction_model = centernet(input_shape=input_shape, num_classes=len(classes), backbone=backbone, max_objects=max_detections, mode="train") ## output shape : topk_x1, topk_y1, topk_x2, topk_y2, scores, class_ids
    prediction_model.load_weights(ckpt_path, by_name=True, skip_mismatch=True)

    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, (512, 512))
    input_tensor = np.expand_dims(resized_image, axis=0)
    result = prediction_model.predict(input_tensor)[0]

    for res in result:
        xmin, ymin, xmax, ymax, score, class_id = int(res[0]), int(res[1]), int(res[2]), int(res[3]), res[4], int(res[5])
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=(0, 0, 255))