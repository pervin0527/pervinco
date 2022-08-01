import os
import cv2
import numpy as np
import tensorflow as tf
from glob import glob


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


def get_detect_result(image):
    image = cv2.resize(image, (detection_shape[0], detection_shape[1]))
    input_tensor = np.expand_dims(image, axis=0)
    detection = detection_model(input_tensor)

    bboxes = detection[0][0].numpy()
    scores = detection[1][0].numpy()

    indexes = np.where(scores > 0.7)[0]
    bboxes = bboxes[indexes]
    
    return bboxes


def bbox_refine(xmin, ymin, xmax, ymax):
    target_width, target_height = frame_width, frame_height
    x_scale = target_width / detection_shape[0]
    y_scale = target_height / detection_shape[1]

    xmin = int(np.round(xmin * x_scale))
    ymin = int(np.round(ymin * y_scale))
    xmax = int(np.round(xmax * x_scale))
    ymax = int(np.round(ymax * y_scale))

    return xmin, ymin, xmax, ymax


def inference():
    capture = cv2.VideoCapture(-1)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    
    while cv2.waitKey(33) != ord('q'):
        ret, frame = capture.read()
        bboxes = get_detect_result(frame)

        result_frame = frame.copy()
        for box in bboxes:
            ymin, xmin, ymax, xmax = box.astype(np.int32)
            xmin, ymin, xmax, ymax = bbox_refine(xmin, ymin, xmax, ymax)

            cv2.rectangle(result_frame, (xmin, ymin), (xmax, ymax), (0, 255, 0))

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
            xmax = min(frame_width, xmax)
            ymax = min(frame_height, ymax)

            edx1 = max(0, -xmin)
            edy1 = max(0, -ymin)
            edx2 = max(0, xmax - frame_width)
            edy2 = max(0, ymax - frame_height)

            cropped = frame[ymin : ymax, xmin : xmax]
            pfld_input = cv2.resize(cropped, (landmark_shape[0], landmark_shape[1]))
            pfld_input = np.expand_dims((pfld_input / 255.0), axis=0)
            pfld_input = tf.convert_to_tensor(pfld_input, dtype=tf.float32, name="input_1")
            landmarks = landmark_model(pfld_input)[0].numpy()
            landmarks = landmarks.reshape(-1, 2) * [size, size] - [edx1, edy1]

            for (x, y) in landmarks.astype(np.int32):
                cv2.circle(result_frame, (xmin + x, ymin + y), 2, (255, 255, 0), thickness=-1)

        result_frame = cv2.flip(result_frame, 1)
        cv2.imshow("result", result_frame)

    capture.release()


if __name__ == "__main__":
    classes = ["face"]
    frame_width, frame_height = 1280, 720
    
    landmark_shape = (112, 112, 3)
    landmark_model = tf.saved_model.load("/data/Models/facial_landmark_68pts_aug/saved_model")

    detection_shape = (384, 384, 3)
    detection_model = tf.saved_model.load("/data/Models/WIDER-FACE-300/saved_model")

    inference()