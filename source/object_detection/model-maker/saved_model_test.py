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


def inference(model_path):
    model = tf.saved_model.load(model_path)
    print("Model Loaded")

    detection_result = []
    files = sorted(glob(f"{frame_path}/*"))
    print(len(files))
    for idx in tqdm(range(len(files))):
        try:
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

                detection_result.append([f"{idx:>06}.jpg", class_id, score])
                for b in bbox:
                    ymin, xmin, ymax, xmax = int(b[0]), int(b[1]), int(b[2]), int(b[3])
                    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), thickness=3)
                cv2.imwrite(f"{save_path}/O/{idx:>06}.jpg", image)
            
            else:
                detection_result.append([f"{idx:>06}.jpg", None, None])
                cv2.imwrite(f"{save_path}/X/{idx:>06}.jpg", image)

        except:
            print(f"{file} Broken")
            
    df = pd.DataFrame(detection_result)
    df.to_csv(f"{save_path}/result.csv", index=False, header=["file_name", "class_ids", "scores"])


if __name__ == "__main__":
<<<<<<< HEAD
    pb_path = "/data/Models/NIPA/BR-set0_384-100/saved_model"
=======
    pb_path = "/data/Models/efficientdet_lite/BR-set0_384-100/saved_model"
>>>>>>> 3b45de715aa72bce876c0f1c2757295d77d0b141
    save_path = f"/data/Datasets/BR/eval"
    frame_path = "/home/jun/Pictures"
    input_shape = (384, 384)
    threshold = 0.8

    if not os.path.isdir(save_path):
        os.makedirs(f"{save_path}/O")
        os.makedirs(f"{save_path}/X")

    total_matched = inference(pb_path)