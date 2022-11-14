import os
import cv2
import shutil
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


def preprocessing(path):
    files = sorted(glob(f"{path}/*"))
    for idx in tqdm(range(len(files))):
        file = files[idx]
        image = cv2.imread(file)
        image = cv2.resize(image, input_shape)

        cv2.imwrite(f"{save_path}/images/{idx:>05}.jpg", image)


def inference(model_path, eval_path):
    model = tf.saved_model.load(model_path)
    print("Model Loaded")

    if not os.path.isdir(save_path):
        os.makedirs(f"{save_path}/O")
        os.makedirs(f"{save_path}/X")

    detection_result = []
    files = sorted(glob(f"{eval_path}/*"))
    print(len(files))
    for idx in tqdm(range(len(files))):
        try:
            file = files[idx]
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

                detection_result.append([f"{idx:>06}.jpg", class_id, score])
                for b in bbox:
                    ymin, xmin, ymax, xmax = int(b[0]), int(b[1]), int(b[2]), int(b[3])
                    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), thickness=3)
                cv2.imwrite(f"{save_path}/O/{idx:>06}.jpg", image)
            
            else:
                indices = np.where(scores > 0.15)[0]
                if indices.size > 0:
                    bbox = boxes[indices]
                    score = scores[indices]
                    class_id = class_ids[indices]

                    detection_result.append([f"{idx:>06}.jpg", class_id, score])
                    for b in bbox:
                        ymin, xmin, ymax, xmax = int(b[0]), int(b[1]), int(b[2]), int(b[3])
                        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), thickness=3)
                    cv2.imwrite(f"{save_path}/X/{idx:>06}.jpg", image)
                else:
                    detection_result.append([f"{idx:>06}.jpg", None, None])
                    cv2.imwrite(f"{save_path}/X/{idx:>06}.jpg", image)

        except:
            print(f"{file} Broken")
            
    df = pd.DataFrame(detection_result)
    df.to_csv(f"{save_path}/result.csv", index=False, header=["file_name", "class_ids", "scores"])


def spot_inference(model_path, eval_path):
    model = tf.saved_model.load(model_path)
    print("Model Loaded")

    detection_result = []
    folders = sorted(glob(f"{eval_path}/*"))
    for folder in folders:
        spot_name = folder.split('/')[-1]
        frames = sorted(glob(f"{folder}/*.jpg"))

        if not os.path.isdir(f"{save_path}/{spot_name}"):
            os.makedirs(f"{save_path}/{spot_name}/O")
            os.makedirs(f"{save_path}/{spot_name}/X")

        for idx in tqdm(range(len(frames))):
            try:
                frame = frames[idx]
                image = cv2.imread(frame)
                input_tensor = np.expand_dims(image, axis=0)

                prediction = model(input_tensor)
                boxes, scores, class_ids = prediction[0][0].numpy(), prediction[1][0].numpy(), prediction[2][0].numpy()

                indices = np.where(scores > threshold)[0]
                if indices.size > 0:
                    bbox = boxes[indices]
                    score = scores[indices]
                    class_id = class_ids[indices]

                    detection_result.append([f"{spot_name}", f"{idx:>06}.jpg", class_id, score])
                    for b in bbox:
                        ymin, xmin, ymax, xmax = int(b[0]), int(b[1]), int(b[2]), int(b[3])
                        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), thickness=3)
                    cv2.imwrite(f"{save_path}/{spot_name}/O/{idx:>06}.jpg", image)
                
                else:
                    indices = np.where(scores > 0.15)[0]
                    if indices.size > 0:
                        bbox = boxes[indices]
                        score = scores[indices]
                        class_id = class_ids[indices]

                        detection_result.append([f"{spot_name}", f"{idx:>06}.jpg", class_id, score])
                        for b in bbox:
                            ymin, xmin, ymax, xmax = int(b[0]), int(b[1]), int(b[2]), int(b[3])
                            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), thickness=3)
                        cv2.imwrite(f"{save_path}/{spot_name}/X/{idx:>06}.jpg", image)
                    else:
                        detection_result.append([f"{spot_name}", f"{idx:>06}.jpg", None, None])
                        cv2.imwrite(f"{save_path}/{spot_name}/X/{idx:>06}.jpg", image)

            except:
                print(f"{frame} Broken")
                
    df = pd.DataFrame(detection_result)
    df.to_csv(f"{save_path}/result.csv", index=False, header=["file_name", "class_ids", "scores"])

if __name__ == "__main__":
    pb_path = "/data/Models/NIPA/BR-set1-300/saved_model"
    frame_path = "/data/Datasets/BR/frames"
    save_path = f"/data/Datasets/BR/eval"
    input_shape = (384, 384)
    threshold = 0.4

    # inference(pb_path, frame_path)
    spot_inference(pb_path, frame_path)
    