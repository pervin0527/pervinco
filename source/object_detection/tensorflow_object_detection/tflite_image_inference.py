import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import tflite_runtime.interpreter as tflite
from glob import glob
from tqdm import tqdm

if __name__ == "__main__":
    model_path = "/data/Models/efficientdet_lite/full-name13-GAP6-300/full-name13-GAP6-300.tflite"
    images_path = "/data/Datasets/SPC/full-name14/test/images"
    label_path = "/data/Datasets/SPC/Labels/labels.txt"
    threshold = 0.7

    LABEL_FILE = pd.read_csv(label_path, sep=' ', index_col=False, header=None)
    CLASSES = LABEL_FILE[0].tolist()

    images = sorted(glob(f"{images_path}/*"))

    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # print_info(input_details, "Input")
    # print_info(output_details, "Output")

    input_shape = input_details[0].get('shape')
    input_width, input_height = input_shape[1], input_shape[2]
    input_dtype = input_details[0].get('dtype')

    total = []
    for idx in tqdm(range(len(images))):
        image_file = images[idx]
        image = cv2.imread(image_file)
        image = cv2.resize(image, (input_height, input_width))
        input_tensor = np.expand_dims(image, axis=0)
                
        interpreter.set_tensor(input_details[0]['index'], input_tensor.astype(np.uint8))
        # interpreter.set_tensor(input_details[0]['index'], input_tensor.numpy().astype(np.float32))
        interpreter.invoke()

        bboxes = interpreter.get_tensor(output_details[0]['index'])
        classes = interpreter.get_tensor(output_details[1]['index'])
        scores = interpreter.get_tensor(output_details[2]['index'])
        num_detections = interpreter.get_tensor(output_details[3]['index'])

        result = [image_file]
        if any(scores[0] > threshold):
            for i, score in enumerate(scores[0]):
                if score > threshold:
                    label_number = classes[0][i]
                    bbox = bboxes[0][i]

                    score = f"{(score * 100):.2f}"
                    label = CLASSES[int(label_number)]
                    ymin, xmin, ymax, xmax = int(bbox[0] * input_width), int(bbox[1] * input_width), int(bbox[2] * input_width), int(bbox[3] * input_width)
                    result.extend([label, score, xmin, ymin, xmax, ymax])
        else:
            result.extend(["No object", -1, -1, -1, -1, -1])

        total.append(result)
     
    df = pd.DataFrame(total)
    df.to_csv('/data/Datasets/SPC/full-name14/test-result.csv', index=False, header=['file name', 'label', 'score', 'xmin', 'ymin', 'xmax', 'ymax'])