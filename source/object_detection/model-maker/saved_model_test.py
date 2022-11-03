import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from glob import glob
from tqdm import tqdm

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


# def preprocess_input(image):
#     image = image.astype(np.float32)
#     image /= 255.0
#     image -= np.array([0.485, 0.456, 0.406])
#     image /= np.array([0.229, 0.224, 0.225])

#     return image


def inference(dataset_path, model_path):
    model = tf.saved_model.load(model_path)
    print("Model Loaded")

    folders = sorted(glob(f"{dataset_path}/*"))
    detection_result = []
    for folder in folders:
        print(folder)
        spot_name = folder.split('/')[-1].split('.')[0]
        frames = sorted(glob(f"{folder}/*.jpg"))

        acc = 0
        for index in tqdm(range(len(frames))):
            image = cv2.imread(frames[index])
            image = cv2.resize(image, input_shape)
            input_tensor = np.expand_dims(image, axis=0)

            prediction = model(input_tensor)
            boxes, scores, class_ids = prediction[0][0].numpy(), prediction[1][0].numpy(), prediction[2][0].numpy()

            indices = np.where(scores > threshold)[0]
            if indices.size > 0:
                bbox = boxes[indices]
                score = scores[indices]
                class_id = class_ids[indices]

                if len(class_id) == 1:
                    if class_id == 1:
                        detection_result.append([f"{spot_name}", f"{index:>06}.jpg", "O"])
                        acc+=1
                    else:
                        detection_result.append([f"{spot_name}", f"{index:>06}.jpg", "X"])
                
                elif len(class_id) > 1:
                    if 1 in class_id:
                        detection_result.append([f"{spot_name}", f"{index:>06}.jpg", "O"])
                        acc += 1
                    else:
                        detection_result.append([f"{spot_name}", f"{index:>06}.jpg", "X"])
            else:
                detection_result.append([f"{spot_name}", f"{index:>06}.jpg", "X"])
            
        print(acc, acc / len(frames) * 100)

    df = pd.DataFrame(detection_result)
    df.to_csv("/data/result.csv", index=False, header=["spot_name", "filename", "is_correct"])


def inference2(dataset_path, model_path):
    model = tf.saved_model.load(model_path)
    print("Model Loaded")

    detection_result = []
    files = sorted(glob(f"{frame_path}/*/JPEGImages/*"))
    print(len(files))
    for idx in tqdm(range(len(files))):
        file = files[idx]
        file_name = file.split('/')[-1].split('.')[0]
        image = cv2.imread(file)
        image = cv2.resize(image, input_shape)
        input_tensor = np.expand_dims(image, axis=0)
        
        prediction = model(input_tensor)
        boxes, scores, class_ids = prediction[0][0].numpy(), prediction[1][0].numpy(), prediction[2][0].numpy()

        indices = np.where(scores > threshold)[0]
        if indices.size > 0:
            bbox = boxes[indices]
            score = scores[indices]
            class_id = class_ids[indices]

            if len(class_id) == 1:
                if class_id == 1:
                    detection_result.append([f"{file_name}", "O"])
                else:
                    detection_result.append([f"{file_name}", "X"])
            
            elif len(class_id) > 1:
                if 1 in class_id:
                    detection_result.append([f"{file_name}", "O"])
                else:
                    detection_result.append([f"{file_name}", "X"])
        else:
            detection_result.append([f"{file_name}", "X"])
            
    df = pd.DataFrame(detection_result)
    df.to_csv("/data/result.csv", index=False, header=["file_name", "is_correct"])


if __name__ == "__main__":
    pb_path = "/data/Models/efficientdet_lite/BR-set2-100/saved_model"
    frame_path = "/data/Datasets/SPC/Cvat/Baskin_robbins"
    input_shape = (384, 384)
    threshold = 0.4

    mean_rgb = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    stddev_rgb = [0.229 * 255, 0.224 * 255, 0.225 * 255]

    total_matched = inference2(frame_path, pb_path)