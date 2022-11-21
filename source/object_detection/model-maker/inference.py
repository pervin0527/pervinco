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


def get_result(model, input_tensor):
    prediction = model(input_tensor)
    bboxes, scores, classes = prediction[0][0].numpy(), prediction[1][0].numpy(), prediction[2][0].numpy()    
    
    result_boxes, result_scores, result_labels = [], [], []
    for i, score in enumerate(scores[:max_detections]):
        if score >= threshold:
            class_id = labels[int(classes[i])-1]
            ymin, xmin, ymax, xmax = int(bboxes[i][0]), int(bboxes[i][1]), int(bboxes[i][2]), int(bboxes[i][3])
            
            result_boxes.append([xmin, ymin, xmax, ymax])
            result_scores.append(score)
            result_labels.append(class_id)

        elif 0.5 <= score < threshold:
            class_id = labels[int(classes[i])-1]
            ymin, xmin, ymax, xmax = int(bboxes[i][0]), int(bboxes[i][1]), int(bboxes[i][2]), int(bboxes[i][3])
            
            result_boxes.append([xmin, ymin, xmax, ymax])
            result_scores.append(score)
            result_labels.append(class_id)

    return result_boxes,result_scores, result_labels


def inference(weight, testset, mode="images"):
    model = tf.saved_model.load(f"{weight}/saved_model")
    print("Model Loaded")

    if mode == "images":
        total_results = []
        if not os.path.isdir(save_dir):
            os.makedirs(f"{save_dir}/O")
            os.makedirs(f"{save_dir}/X")

        test_images = sorted(glob(f"{testset}/*"))
        print("total test images : ", len(test_images))

        for idx in tqdm(range(len(test_images))):
            test_image = test_images[idx]
            name = test_image.split('/')[-1]

            try:
                image = cv2.imread(test_images[idx])
                image = cv2.resize(image, input_shape)
                input_tensor = np.expand_dims(image, axis=0)

            except:
                print(f"File {test_images[idx]} is Broken")
                continue

            res_boxes, res_scores, res_labels = get_result(model, input_tensor)
            if len(res_boxes) > 0:
                total_results.append([name, res_labels, res_scores])
                for box in res_boxes:
                    cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 3)
                cv2.imwrite(f"{save_dir}/O/{idx:06}.jpg", image)
            else:
                total_results.append([name, None, None])
                cv2.imwrite(f"{save_dir}/X/{idx:06}.jpg", image)

        df = pd.DataFrame(total_results)
        df.to_csv(f"{save_dir}/result.csv", index=False, header=["filename", "labels", "scores"])

    elif mode == "folders":
        folders = sorted(glob(f"{testset}/*"))
        # folders = [f"{testset}/삼청마당_18", f"{testset}/삼청마당_15", f"{testset}/서초우성_09"]
        for folder in folders:
            total_results = []
            test_images = sorted(glob(f"{folder}/*"))
            print(f"{folder} has {len(test_images)} images")

            spot = folder.split('/')[-1]
            if not os.path.isdir(f"{save_dir}/{spot}"):
                os.makedirs(f"{save_dir}/{spot}/O")
                os.makedirs(f"{save_dir}/{spot}/X")

            for idx in tqdm(range(len(test_images))):
                test_image = test_images[idx]
                name = test_image.split('/')[-1]
            
                try:
                    image = cv2.imread(test_images[idx])
                    image = cv2.resize(image, input_shape)
                    input_tensor = np.expand_dims(image, axis=0)

                except:
                    print(f"File {test_images[idx]} is Broken")
                    continue

                res_boxes, res_scores, res_labels = get_result(model, input_tensor)
                if len(res_boxes) > 0:
                    total_results.append([name, res_labels, res_scores])
                    for box in res_boxes:
                        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 3)
                    cv2.imwrite(f"{save_dir}/{spot}/O/{idx:06}.jpg", image)
                else:
                    total_results.append([name, None, None])
                    cv2.imwrite(f"{save_dir}/{spot}/X/{idx:06}.jpg", image)
    
            df = pd.DataFrame(total_results)
            df.to_csv(f"{save_dir}/{spot}.csv", index=False, header=["filename", "labels", "scores"])


if __name__ == "__main__":
    weight_dir = "/home/ubuntu/Models/efficientdet_lite/BR-set1-300"
    label_dir = "/home/ubuntu/Datasets/BR/Labels/labels.txt"
    # testset_dir = "/home/ubuntu/Datasets/BR/testset/set1"
    testset_dir = "/home/ubuntu/Datasets/BR/frames"
    mode = "folders"

    input_shape = (384, 384)
    max_detections = 10
    threshold = 0.9

    model_name = weight_dir.split("/")[-1]
    testset_name = testset_dir.split("/")[-1]
    save_dir = f"/home/ubuntu/Datasets/BR/testset/{testset_name}-{model_name}"

    labels = pd.read_csv(label_dir, sep=',', index_col=False, header=None)
    labels = labels[0].tolist()

    inference(weight_dir, testset_dir, mode)