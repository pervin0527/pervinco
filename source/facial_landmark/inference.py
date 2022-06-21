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
    image = np.array(image).astype(np.uint8)

    for (x, y) in landmarks:
        cv2.circle(image, (int(x), int(y)), radius=1, color=(0, 0, 255), thickness=-1)

    return image


def inference(video_capture_index):
    capture = cv2.VideoCapture(video_capture_index)
    
    while cv2.waitKey(33) != ord('q'):
        ret, frame = capture.read()
        frame = cv2.resize(frame, (512, 512))
        height, width = frame.shape[:2]

        image = cv2.resize(frame, (input_shape[0], input_shape[1]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image / 255.0
        image = np.expand_dims(image, axis=0)

        prediction = model.predict(image)
        pred_landmarks = prediction * input_shape[0]
        pred_landmarks[0::2] = pred_landmarks[0::2] * width / input_shape[0]
        pred_landmarks[1::2] = pred_landmarks[1::2] * height / input_shape[0]
        pred_landmarks = pred_landmarks.reshape(-1, 2)

        result_image = get_overlay(index=0, image=frame, landmarks=pred_landmarks)
        cv2.imshow("result", result_image)

    capture.release()
    cv2.destroyAllWindows()


def model_load(model_path):
    model = PFLDInference(inputs=input_shape, is_train=False)
    model.load_weights(model_path, by_name=True)

    return model

if __name__ == "__main__":
    input_shape = [112, 112, 3]
    ckpt_path = "/data/Models/facial_landmark/best.h5"
    
    model = model_load(ckpt_path)
    inference("/data/Datasets/WFLW/inference_data/test01.mp4") # or -1
