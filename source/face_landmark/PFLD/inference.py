import os
import cv2
import numpy as np
import tensorflow as tf

from glob import glob
from model import PFLD

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


def yolo2voc(class_id, width, height, x, y, w, h):
    xmin = int((x * width) - (w * width) / 2.0)
    ymin = int((y * height) - (h * height) / 2.0)
    xmax = int((x * width) + (w * width) / 2.0)
    ymax = int((y * height) + (h * height) / 2.0)
    class_id = int(class_id)

    return (class_id, xmin, ymin, xmax, ymax)


def image_inference(dir):
    image_files = sorted(glob(f"{dir}/imgs/*.jpg"))
    label_files = sorted(glob(f"{dir}/labels/*.txt"))

    for image_file, label_file in zip(image_files, label_files):
        image = cv2.imread(image_file)
        height, width = image.shape[0], image.shape[1]
        
        labels = open(label_file, "r").readlines()
        labels = labels[0].split()
        label, x, y, w, h = int(labels[0]), float(labels[1]), float(labels[2]), float(labels[3]), float(labels[4])
        
        label, xmin, ymin, xmax, ymax = yolo2voc(label, width, height, x, y, w, h)

        result_image = image.copy()
        cv2.rectangle(result_image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color=(0, 0, 255))

        w = xmax - xmin + 1
        h = ymax - ymin + 1
        cx = xmin + w // 2
        cy = ymin + h // 2

        size = int(max([w, h]) * 1.1)
        xmin = cx - size // 2
        xmax = xmin + size
        ymin = cy - size //2 
        ymax = ymin + size

        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(width, xmax)
        ymax = min(height, ymax)

        edx1 = max(0, -xmin)
        edy1 = max(0, -ymin)
        edx2 = max(0, xmax - width)
        edy2 = max(0, ymax - height)

        cropped = image[ymin : ymax, xmin : xmax]
        pfld_input = cv2.resize(cropped, (input_shape[0], input_shape[1]))
        pfld_input = np.expand_dims((pfld_input / 255.0), axis=0)
        landmarks = model.predict(pfld_input)
        landmarks = landmarks[0].reshape(-1, 2) * [size, size] - [edx1, edy1]

        for (x, y) in landmarks.astype(np.int32):
            cv2.circle(result_image, (xmin + x, ymin + y), 2, (0, 255, 255))

        cv2.imshow("result", result_image)
        cv2.waitKey(0)


def model_load(model_path):
    model = PFLD()
    model.built = True
    model.load_weights(model_path, by_name=True, skip_mismatch=True)
    model.summary()

    return model


if __name__ == "__main__":
    mode = "images"
    input_shape = [112, 112, 3]
    ckpt_path = "/data/Models/facial_landmark_68pts/pfld.h5"
    
    model = model_load(ckpt_path)

    if mode == "images":
        data_dir = "/data/Datasets/300VW_Dataset_2015_12_14/original/001"
        image_inference(data_dir)
