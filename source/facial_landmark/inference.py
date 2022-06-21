import os
import cv2
import numpy as np
import tensorflow as tf

from glob import glob
from model import PFLDInference

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


def get_overlay(index, image, landmarks):
    for (x, y) in landmarks:
        cv2.circle(image, (int(x), int(y)), radius=1, color=(0, 0, 255), thickness=-1)

    return image


def video_inference(video_capture_index):
    capture = cv2.VideoCapture(video_capture_index)

    while cv2.waitKey(33) != ord('q'):
        ret, frame = capture.read()
        height, width = frame.shape[:2]
        # height, width = input_shape[0], input_shape[1]

        resized_image = cv2.resize(frame, (input_shape[0], input_shape[1]))
        input_tensor = np.expand_dims(resized_image, axis=0)
        input_tensor = input_tensor / 255.0

        prediction = model.predict(input_tensor)
        pred_landmarks = prediction * input_shape[0]
        pred_landmarks[0::2] = pred_landmarks[0::2] * width / input_shape[0]
        pred_landmarks[1::2] = pred_landmarks[1::2] * height / input_shape[0]
        pred_landmarks = pred_landmarks.reshape(-1, 2)

        # result_image = resized_image.copy()
        # result_image = get_overlay(index=0, image=result_image, landmarks=pred_landmarks)
        result_image = get_overlay(index=0, image=frame, landmarks=pred_landmarks)
        cv2.imshow("result", result_image)

    capture.release()
    cv2.destroyAllWindows()


def image_inference(image_file):
    image = cv2.imread(image_file)
    height, width = image.shape[:2]
    resized_image = cv2.resize(image, (input_shape[0], input_shape[1]))
    input_tensor = np.expand_dims(resized_image, axis=0)
    input_tensor = input_tensor / 255.0

    pred = model.predict(input_tensor)
    pred = pred * input_shape[0]
    pred[0::2] = pred[0::2] * width / input_shape[0]
    pred[1::2] = pred[1::2] * height / input_shape[0]
    pred = pred.reshape(-1, 2)

    result_image = get_overlay(index=0, image=image, landmarks=pred)
    cv2.imshow("result", result_image)
    cv2.waitKey(0)


def model_load(model_path):
    model = PFLDInference(inputs=input_shape, is_train=False)
    model.load_weights(model_path, by_name=True)

    return model

if __name__ == "__main__":
    input_shape = [112, 112, 3]
    ckpt_path = "/data/Models/facial_landmark/best.h5"
    
    model = model_load(ckpt_path)
    # video_inference("/data/Datasets/WFLW/inference_data/test01.mp4")
    image_inference("/data/Datasets/WFLW/inference_data/1.png")
