import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import numpy as np
import tensorflow as tf
from glob import glob
from random import randint
from model_backup import PFLDInference

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


def read_pts(filename):
    return np.loadtxt(filename, comments=("version:", "n_points:", "{", "}"))


def generate_bbox(keypoints, frame):
    xy = np.min(keypoints, axis=0).astype(np.int32)
    zz = np.max(keypoints, axis=0).astype(np.int32)
    wh = zz - xy + 1

    frame_h, frame_w = frame.shape[:2]

    center = (xy + wh/2).astype(np.int32)
    size = int(np.max(wh)*1.2)
    xy = center - size//2
    x1, y1 = xy
    x2, y2 = xy + size

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame_w, x2)
    y2 = min(frame_h, y2)

    edx1 = max(0, -x1)
    edy1 = max(0, -y1)
    edx2 = max(0, x2 - frame_w)
    edy2 = max(0, y2 - frame_h)

    crop_image = frame[y1:y2, x1:x2]
    if (edx1 > 0 or edy1 > 0 or edx2 > 0 or edy2 > 0):
        crop_image = cv2.copyMakeBorder(crop_image, edy1, edy2, edx1, edx2, cv2.BORDER_CONSTANT, 0)

    return size, (edx1, edy1), (x1, y1), crop_image


def inference(folder_list, model):
    random_number = randint(0, len(folder_list))
    folder = folder_list[random_number]

    video_file = f"{folder}/vid.avi"
    pts_files = sorted(glob(f"{folder}/annot/*.pts"))

    index = 0
    capture = cv2.VideoCapture(video_file)

    while cv2.waitKey(33) != ord('q'):
        _, frame = capture.read()
        height, width = frame.shape[:2]

        points = read_pts(pts_files[index])
        size, (edx1, edy1), (x1, y1), crop_image = generate_bbox(points, frame)

        input = cv2.resize(crop_image, (112, 112))
        input = np.expand_dims(input, axis=0)
        
        pred = model.predict(input, verbose=0)
        pred_landmark = pred[0]
        pred_landmark = pred_landmark.reshape(-1, 2) * [size, size] - [edx1, edy1]

        for (x, y) in pred_landmark.astype(np.int32):
            cv2.circle(frame, (x1+x, y1+y), 2, (0, 0, 255), thickness=-1)

        cv2.imshow("result", frame)


def load_model(model_path):
    model = PFLDInference(inputs=[112, 112, 3], is_train=False, keypoints=68*2)
    model.load_weights(model_path, by_name=True)

    return model


if __name__ == "__main__":
    CKPT_DIR = "/data/Models/face_landmark_68pts/007569.h5"
    DATA_DIR = "/data/Datasets/300VW_Dataset_2015_12_14/original"
    folders = sorted(glob(f"{DATA_DIR}/*"))

    model = load_model(CKPT_DIR)
    inference(folders, model)