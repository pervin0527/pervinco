import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from glob import glob
from tqdm import tqdm

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

def print_info(info_list, type):
    print(f"############################################# Model {type} info #############################################")
    for info in info_list:
        print(info, "\n")
    print("############################################################################################################## \n")

def image_preprocess(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image)
    image = tf.image.resize(image, (input_width, input_height))
    input_tensor = tf.expand_dims(image, axis=0)
    
    return input_tensor

if __name__ == "__main__":
    model_path = "/data/Models/classification/SPC/2022.03.24_18:18/test_metadata.tflite"
    images_path = "/data/test/spc-sample"
    label_path = "/data/Datasets/SPC/Labels/labels.txt"
    threshold = 0.7

    LABEL_FILE = pd.read_csv(label_path, sep=' ', index_col=False, header=None)
    CLASSES = LABEL_FILE[0].tolist()
    CLASSES.insert(0, "Backgroud")

    images = sorted(glob(f"{images_path}/*"))

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    os.system("clear")
    print_info(input_details, "Input")
    print_info(output_details, "Output")

    input_shape = input_details[0].get('shape')
    input_width, input_height = input_shape[1], input_shape[2]
    input_dtype = input_details[0].get('dtype')

    for idx in tqdm(range(len(images))):
        image_path = images[idx]
        input_tensor = image_preprocess(image_path)
        interpreter.set_tensor(input_details[0]['index'], input_tensor.numpy().astype(np.uint8))
        # interpreter.set_tensor(input_details[0]['index'], input_tensor.numpy().astype(np.float32))
        interpreter.invoke()

        result = interpreter.get_tensor(output_details[0]['index'])[0]
        
        idx = np.argmax(result)
        score = result[idx]
        label = CLASSES[idx]

        image = cv2.imread(image_path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (input_width, input_height))

        cv2.putText(image, f"{label} {score:.2f}%", (20, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255))
        image = cv2.resize(image, (512, 512))
        cv2.imshow('result', image)
        cv2.waitKey(0)    