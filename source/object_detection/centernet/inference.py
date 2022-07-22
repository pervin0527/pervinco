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


def preprocess_image(image):
    image = image.astype(np.float32)

    image[..., 0] -= 103.939
    image[..., 1] -= 116.779
    image[..., 2] -= 123.68

    return image


def draw_result(idx, image, detections):
    scores = detections[:, 4]
    indices = np.where(scores > 0.7)[0]
    print(indices, detections[indices])
    detections[:, [0, 2]] = np.clip(detections[:, [0, 2]], 0, image.shape[1])
    detections[:, [1, 3]] = np.clip(detections[:, [1, 3]], 0, image.shape[0])

    if len(indices):
        result_image = image.copy()
        for result in detections[indices]:
            xmin, ymin, xmax, ymax, score, label = int(result[0]), int(result[1]), int(result[2]), int(result[3]), result[4], int(result[5])
            cv2.rectangle(result_image, (xmin, ymin), (xmax, ymax), (0, 0, 255))

        cv2.imwrite("./epoch_end/inference.jpg", result_image)


if __name__ == "__main__":
    img_path = "./samples/sample01.jpg"
    ckpt_path = "/home/ubuntu/Models/CenterNet/custom_unfreeze.h5"
    
    input_shape = (512, 512, 3)
    backbone = "resnet50"
    classes = ["face"]
    max_detections = 30
    freeze_backbone = False

    model, pred_model, _ = centernet(input_shape=input_shape, num_classes=len(classes), backbone=backbone, max_detections=max_detections, mode="train", freeze_bn=freeze_backbone)
    model.load_weights(ckpt_path, by_name=True, skip_mismatch=True)


    image = cv2.imread(img_path)
    input_image = cv2.resize(image, (input_shape[0], input_shape[1]))
    input_image = preprocess_image(input_image)
    input_image = np.expand_dims(input_image, axis=0)
    
    detections = pred_model.predict(input_image, verbose=0)[0]

    result_image = cv2.resize(image, (input_shape[0] // 4, input_shape[1] // 4))
    draw_result(0, result_image, detections)