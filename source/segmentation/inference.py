import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import advisor
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import tensorflow as tf

from glob import glob
from model import DeepLabV3Plus


def live_stream_inference(height, width):
    capture = cv2.VideoCapture(-1)  
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, height)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, width)
    
    while cv2.waitKey(33) != ord('q'):
        ret, frame = capture.read()
        
        image = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        input_tensor = np.expand_dims(image, axis=0)
        
        prediction = model.predict(input_tensor)
        
        if prediction.shape[:-1] != IMG_SIZE:
            prediction = np.argmax(prediction[0], axis=-1)

        decoded_mask = decode_segmentation_masks(prediction)
        overlay_image = get_overlay(decoded_mask, cv2.resize(frame, (IMG_SIZE, IMG_SIZE)))

        cv2.imshow("PREDICTION", cv2.resize(overlay_image, (height, width)))

    capture.release()
    cv2.destroyAllWindows()


def image_file_inference(height, width):
    image_files = sorted(glob(f"{IMG_PATH}/*"))
    for image_file in image_files:
        image = cv2.imread(image_file)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        input_tensor = np.expand_dims(image, axis=0)

        prediction = model.predict(input_tensor)

        if prediction.shape[:-1] != IMG_SIZE:
            prediction = np.argmax(prediction[0], axis=-1)

        decoded_mask = decode_segmentation_masks(prediction)
        overlay_image = get_overlay(decoded_mask, image)

        result_image = cv2.resize(overlay_image, (height, width))
        cv2.imshow("PREDICTION", result_image)
        cv2.waitKey(0)


def load_model_with_ckpt(ckpt_path, include_infer=False):
    # dice_loss = advisor.losses.DiceLoss()
    # categorical_focal_loss = advisor.losses.CategoricalFocalLoss()
    # loss = dice_loss + (1 * categorical_focal_loss)
    
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    metrics = tf.keras.metrics.OneHotMeanIoU(num_classes=len(COLORMAP))
    optimizer = tf.keras.optimizers.Adam()

    trained_model = DeepLabV3Plus(IMG_SIZE, IMG_SIZE, len(COLORMAP), backbone_name=BACKBONE_NAME, backbone_trainable=BACKBONE_TRAINABLE, final_activation=FINAL_ACTIVATION)
    trained_model.load_weights(ckpt_path)

    if include_infer:
        inference = tf.keras.layers.Lambda(lambda x : tf.argmax(tf.squeeze(x, axis=0), axis=-1))(trained_model.output)
        model = tf.keras.Model(inputs=trained_model.input, outputs=inference)
    
        model.compile(optimizer=optimizer, loss=loss, metrics=[metrics])
        return model

    else:
        trained_model.compile(optimizer=optimizer, loss=loss, metrics=[metrics])
        return trained_model


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
    CKPT_PATH = "/data/Models/segmentation/UNITY-ResNet101-AUGMENT_50/best.ckpt"
    IMG_PATH = "/data/Datasets/VOCdevkit/VOC2012/BASIC/valid/images"
    INFERENCE = "video"

    IMG_SIZE = 320
    BACKBONE_NAME = CKPT_PATH.split('/')[-2].split('-')[1]
    print(BACKBONE_NAME)
    BACKBONE_TRAINABLE = False
    FINAL_ACTIVATION =  None
    INCLUDE_INFER = False
    OUTPUT_SIZE = 320, 320

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

    model = load_model_with_ckpt(CKPT_PATH, INCLUDE_INFER)
    model.summary()
    
    if INFERENCE.lower() == "video":
        live_stream_inference(OUTPUT_SIZE[0], OUTPUT_SIZE[1])

    else:
        image_file_inference(OUTPUT_SIZE[0], OUTPUT_SIZE[1])

