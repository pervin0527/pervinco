import os
import cv2
import numpy as np
import tensorflow as tf
from glob import glob
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


def preprocess_image(image):
    image = image.astype(np.float32)

    image[..., 0] -= 103.939
    image[..., 1] -= 116.779
    image[..., 2] -= 123.68

    return image


def draw_result(idx, image, detections):
    scores = detections[:, 4]
    indices = np.where(scores > 0.55)[0]
    detections[:, [0, 2]] = np.clip(detections[:, [0, 2]], 0, image.shape[1])
    detections[:, [1, 3]] = np.clip(detections[:, [1, 3]], 0, image.shape[0])

    target_size = input_shape[0]
    x_, y_ = input_shape[0] // 4, input_shape[1] // 4

    x_scale = target_size / x_
    y_scale = target_size / y_
    result_image = image.copy()
    if len(indices):
        for result in detections[indices]:
            xmin, ymin, xmax, ymax, score, label = int(result[0]), int(result[1]), int(result[2]), int(result[3]), result[4], int(result[5])

            xmin = int(np.round(xmin * x_scale))
            ymin = int(np.round(ymin * y_scale))
            xmax = int(np.round(xmax * x_scale))
            ymax = int(np.round(ymax * y_scale))

            cv2.rectangle(result_image, (xmin, ymin), (xmax, ymax), (0, 0, 255))

    return result_image


def inference_images(image_dir):
    image_files = sorted(glob(f"{image_dir}/*"))

    for index, image_file in enumerate(image_files):
        image = cv2.imread(image_file)
        resized_image = cv2.resize(image, (input_shape[0], input_shape[1]))
        input_tensor = preprocess_image(resized_image)
        input_tensor = np.expand_dims(input_tensor, axis=0)

        detection_result = pred_model.predict(input_tensor, verbose=0)[0]
        result_image = draw_result(index, resized_image, detection_result)

        cv2.imshow("detection_result", result_image)
        cv2.waitKey(0)


def inference_frames(vid_index):
    capture = cv2.VideoCapture(vid_index)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    capture.set(cv2.CAP_PROP_BRIGHTNESS, 100)
    capture.set(cv2.CAP_PROP_AUTOFOCUS, 0)

    index = 0
    while cv2.waitKey(33) != ord('q'):
        ret, frame = capture.read()
        height, width = frame.shape[:2]

        resized_frame = cv2.resize(frame, (input_shape[0], input_shape[1]))
        input_tensor = preprocess_image(resized_frame)
        input_tensor = np.expand_dims(input_tensor, axis=0)

        detection_result = pred_model.predict(input_tensor, verbose=0)[0]
        result_image = draw_result(index, resized_frame, detection_result)
        index += 1

        cv2.imshow("detection_result", result_image)


if __name__ == "__main__":
    img_path = "/data/Datasets/300VW_Dataset_2015_12_14/original/001/imgs"
    ckpt_path = "/data/Models/CenterNet/custom_unfreeze.h5"
    
    input_shape = (512, 512, 3)
    backbone = "resnet50"
    classes = ["face"]
    max_detections = 30
    freeze_backbone = False

    pred_model = centernet(input_shape=input_shape, num_classes=len(classes), backbone=backbone, max_detections=max_detections, mode="predict", freeze_bn=freeze_backbone)
    pred_model.load_weights(ckpt_path, by_name=True, skip_mismatch=True)
    pred_model.summary()

    inference_frames(-1)
    # inference_images(img_path)