import os
import sys
import cv2
import yaml
import timeit
import numpy as np
import tensorflow as tf

from glob import glob
from model import DeepLabV3Plus
from metrics import Sparse_MeanIoU

# np.set_printoptions(threshold=sys.maxsize)
# GPU setup
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
        

def live_stream_inference(height, width, file=None):
    if file == None:
        file = -1    
    
    capture = cv2.VideoCapture(file)  
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, width)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, height)
    
    while cv2.waitKey(33) != ord('q'):
        ret, frame = capture.read()
        frame = cv2.flip(frame, 0)
        frame = cv2.flip(frame, 1)

        start_time = timeit.default_timer()
        
        image = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = image / 127.5 - 1
        input_tensor = np.expand_dims(image, axis=0)
        
        prediction = model.predict(input_tensor, verbose=0)
        prediction = np.argmax(prediction[0], axis=-1)
        decoded_mask = decode_segmentation_masks(prediction)
        overlay_image = get_overlay(decoded_mask, cv2.resize(frame, (IMG_SIZE, IMG_SIZE)))

        end_time = timeit.default_timer()
        fps = int(1./(end_time - start_time))

        result_image = cv2.resize(overlay_image, (height, width))
        cv2.putText(result_image, text=f"FPS : {fps}", org=(10, 100), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=3, color=(255, 255, 255), thickness=3)

        cv2.imshow("PREDICTION", result_image)
        # cv2.imshow("frame", frame)

    capture.release()
    cv2.destroyAllWindows()


def image_file_inference(height, width):
    image_files = sorted(glob(f"{img_dir}/*.jpg"))
    for image_file in image_files:
        image = cv2.imread(image_file)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        img = image.copy()
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = image / 127.5 - 1
        input_tensor = np.expand_dims(image, axis=0)

        prediction = model.predict(input_tensor, verbose=0)
        prediction = np.argmax(prediction[0], axis=-1)
        decoded_mask = decode_segmentation_masks(prediction)
        overlay_image = get_overlay(decoded_mask, img)

        result_image = cv2.resize(overlay_image, (height, width))
        cv2.imshow("PREDICTION", result_image)
        cv2.waitKey(0)


def load_model_with_ckpt(ckpt_path, include_infer=False):  
    if config["ONE_HOT"]:
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        metrics = tf.keras.metrics.OneHotMeanIoU(num_classes=len(config["CLASSES"]))
    else:
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metrics = Sparse_MeanIoU(num_classes=len(config["CLASSES"]))

    optimizer = tf.keras.optimizers.Adam(learning_rate=config["LR_START"])
    trained_model = DeepLabV3Plus(IMG_SIZE, IMG_SIZE, len(COLORMAP), backbone_name=BACKBONE_NAME, backbone_trainable=False, final_activation=None, original_output=ORIGINAL_OUTPUT)
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
    model_dir = "/data/Models/segmentation/NEW_OUTPUT/VOC2012-AUGMENT_50-ResNet101"
    ckpt = f"{model_dir}/best.ckpt"
    inference_layer = False
    output_shape = 960, 720

    inference_type = "video" # video, images
    video_dir = "/data/test_image/20220706_162517.mp4"
    img_dir = "/data/test_image"

    with open(f"{model_dir}/config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    IMG_SIZE = config["IMG_SIZE"]
    BACKBONE_NAME = config["BACKBONE_NAME"]
    ORIGINAL_OUTPUT = config["ORIGINAL_OUTPUT"]
    COLORMAP = [[0, 0, 0], # background
                [0, 0, 0], # aeroplane
                [0, 0, 0], # bicycle
                [0, 0, 0], # bird
                [0, 0, 0], # boat
                [0, 0, 0], # bottle
                [0, 0, 0], # bus
                [0, 0, 255], # car
                [0, 0, 0], # cat
                [0, 0, 0], # chair
                [0, 0, 0], # cow
                [0, 0, 0], # diningtable
                [0, 0, 0], # dog
                [0, 0, 0], # horse
                [0, 255, 0], # motorbike
                [255, 0, 0], # person
                [0, 0, 0], # potted plant
                [0, 0, 0], # sheep
                [0, 0, 0], # sofa
                [0, 0, 0], # train
                [0, 0, 0] # tv/monitor
    ]
    COLORMAP = np.array(COLORMAP, dtype=np.uint8)

    model = load_model_with_ckpt(ckpt, inference_layer)
    model.summary()
    
    if inference_type.lower() == "video":
        live_stream_inference(output_shape[0], output_shape[1], file=video_dir)

    elif inference_type.lower() == "images":
        image_file_inference(output_shape[0], output_shape[1])

