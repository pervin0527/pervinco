import os
import cv2
import yaml
import numpy as np
import tensorflow as tf
from glob import glob
from models.centernet import centernet
from data.data_utils import read_label_file

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
    config_path = "./configs/train.yaml"
    weight_path = "/home/ubuntu/Models/centernet/unfreeze.h5"
    test_path = "./test_imgs"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    class_names = read_label_file(config["path"]["label_path"])
    num_classes = len(class_names)
    model = centernet([config["train"]["input_shape"][0], config["train"]["input_shape"][1], 3],
                      num_classes,
                      backbone=config["train"]["backbone"],
                      max_objects=config["train"]["max_detection"],
                      mode="predict")

    model.load_weights(weight_path)
    model.summary()

    if not os.path.isdir("./predictions"):
        os.makedirs("./predictions")

    for index, test_img in enumerate(sorted(glob(f"{test_path}/*.jpg"))):
        original_image = cv2.imread(test_img)
        resized_image = cv2.resize(original_image, (config["train"]["input_shape"][0], config["train"]["input_shape"][1]))
        input_tensor = np.expand_dims(resized_image, axis=0)

        prediction = model.predict(input_tensor)[0]

        output_size = config["train"]["input_shape"][0] // 4
        x_scale = config["train"]["input_shape"][0] / output_size
        y_scale = config["train"]["input_shape"][1] / output_size

        result_image = original_image.copy()
        indices = np.where(prediction[:, 4] > config["train"]["threshold"])
        for idx in indices[0]:
            result = prediction[idx]
            bbox = result[0:4]
            score = result[4]
            class_id = result[5]

            bbox[bbox < 0] = 0
            bbox[bbox > config["train"]["input_shape"][0]] = config["train"]["input_shape"][0]

            xmin = int(np.round(bbox[0] * x_scale))
            ymin = int(np.round(bbox[1] * y_scale))
            xmax = int(np.round(bbox[2] * x_scale))
            ymax = int(np.round(bbox[3] * y_scale))
            class_name = class_names[int(class_id)]

            cv2.rectangle(result_image, (xmin, ymin), (xmax, ymax), (0, 0, 255))
            print(class_name, score)

            cv2.imwrite(f"./predictions/pred_{index:>04}.jpg", result_image)