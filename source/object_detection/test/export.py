import os
import cv2
import yaml
import numpy as np
import tensorflow as tf
from models import yolov3, DecodeBox
from utils.utils import get_classes, preprocess_input

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

if __name__ == "__main__":
    with open("./config.yaml", "r") as f:
        config = yaml.safe_load(f)

    class_names, num_classes = get_classes(config["classes_path"])
    print(num_classes, class_names)

    model = yolov3([config["input_shape"][0], config["input_shape"][1], 3], num_classes, config["phi"])
    model.load_weights(config["save_path"] + '/weights.h5')

    outputs = tf.keras.layers.Lambda(DecodeBox, 
                                     output_shape = (1,), 
                                     name = 'yolo_eval',
                                     arguments = {'num_classes':num_classes, 
                                                  'input_shape':config["input_shape"], 
                                                  'confidence':config["score_threshold"], 
                                                  'nms_iou':config["iou_threshold"], 
                                                  'max_boxes':config["max_detections"], 
                                                  'letterbox_image':True})(model.output)
    infer_model = tf.keras.Model(model.input, outputs)
    infer_model.summary()

    image = cv2.imread("./dog.jpg")
    image = cv2.resize(image, (config["input_shape"][0], config["input_shape"][1]))
    input_tensor = preprocess_input(image.astype(np.float32))
    input_tensor = np.expand_dims(input_tensor, axis=0)
    print(input_tensor.shape)

    outputs = infer_model.predict(input_tensor)