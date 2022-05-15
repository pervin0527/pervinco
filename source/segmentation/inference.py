import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from glob import glob
from model import DeepLabV3Plus

def read_image(image, mask=False):
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.image.resize(images=image, size=[IMG_SIZE, IMG_SIZE])
    image.set_shape([IMG_SIZE, IMG_SIZE, 3])

    return image


def infer(model, image_tensor):
    predictions = model.predict(np.expand_dims((image_tensor), axis=0))
    predictions = np.squeeze(predictions)
    predictions = np.argmax(predictions, axis=2)

    return predictions


def decode_segmentation_masks(mask):
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    for l in range(0, len(COLORMAP)):
        idx = mask == l
        r[idx] = COLORMAP[l, 0]
        g[idx] = COLORMAP[l, 1]
        b[idx] = COLORMAP[l, 2]
    rgb = np.stack([r, g, b], axis=2)

    return rgb


def get_overlay(image, colored_mask):
    image = tf.keras.preprocessing.image.array_to_img(image)
    image = np.array(image).astype(np.uint8)
    overlay = cv2.addWeighted(image, 0.35, colored_mask, 0.65, 0)

    return overlay


if __name__ == "__main__":
    CKPT_PATH = "/data/Models/segmentation/VOC2012-ResNet101-AUGMENT_10/best.ckpt"
    IMG_PATH = "/data/Datasets/VOCdevkit/VOC2012/BASIC/valid/images"

    IMG_SIZE = 320
    BACKBONE_NAME = "ResNet101"
    BACKBONE_TRAINABLE = False
    FINAL_ACTIVATION = "softmax"
    COLORMAP = [[0, 0, 0], # background
                [128, 0, 0], # aeroplane
                [0, 128, 0], # bicycle
                [128, 128, 0], # bird
                [0, 0, 128], # boat
                [128, 0, 128], # bottle
                [0, 128, 128], # bus
                [128, 128, 128], # car
                [64, 0, 0], # cat
                [192, 0, 0], # chair
                [64, 128, 0], # cow
                [192, 128, 0], # diningtable
                [64, 0, 128], # dog
                [192, 0, 128], # horse
                [64, 128, 128], # motorbike
                [192, 128, 128], # person
                [0, 64, 0], # potted plant
                [128, 64, 0], # sheep
                [0, 192, 0], # sofa
                [128, 192, 0], # train
                [0, 64, 128] # tv/monitor
    ]
    COLORMAP = np.array(COLORMAP, dtype=np.uint8)

    model = DeepLabV3Plus(IMG_SIZE, IMG_SIZE, len(COLORMAP), backbone_name=BACKBONE_NAME, backbone_trainable=BACKBONE_TRAINABLE, final_activation=FINAL_ACTIVATION)
    model.load_weights(CKPT_PATH)
    model.summary()

    capture = cv2.VideoCapture(-1)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while cv2.waitKey(33) != ord('q'):
        ret, frame = capture.read()
        input_tensor = read_image(frame)
        prediction = infer(model, input_tensor)
        decoded_mask = decode_segmentation_masks(prediction)
        overlay_image = get_overlay(decoded_mask, cv2.resize(frame, (IMG_SIZE, IMG_SIZE)))

        cv2.imshow("PREDICTION", cv2.resize(overlay_image, (640, 480)))

    capture.release()
    cv2.destroyAllWindows()