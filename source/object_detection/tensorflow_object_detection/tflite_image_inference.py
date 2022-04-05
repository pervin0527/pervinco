import os
from tabnanny import check
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import xml.etree.ElementTree as ET
import tflite_runtime.interpreter as tflite
from glob import glob
from tqdm import tqdm
from lxml.etree import Element, SubElement, tostring

def read_xml(xml_file, classes):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    width = int(root.find('size').find('width').text)
    height = int(root.find('size').find('height').text)
    objects = root.findall("object")
    
    # bboxes, labels, areas = [], [], []
    bboxes, labels = [], []
    if len(objects) > 0:
        class_names = [object.findtext("name") for object in objects]
        
        for idx, name in enumerate(class_names):
            if name in classes:
                bbox = objects[idx].find("bndbox")

                xmin = int(float(bbox.find('xmin').text))
                ymin = int(float(bbox.find('ymin').text))
                xmax = int(float(bbox.find('xmax').text))
                ymax = int(float(bbox.find('ymax').text))               
                    
                bboxes.append([xmin, ymin, xmax, ymax])
                labels.append(name)
                # areas.append((xmax - xmin) * (ymax - ymin))

    # return bboxes, labels, areas    
    return bboxes, labels

def draw_result(detection_results):
    record = detection_results[:]
    image_file = record.pop(0)
    file_name = image_file.split('/')[-1].split('.')[0]
    image = cv2.imread(image_file)
    image = cv2.resize(image, (input_height, input_width))

    result_length = int(len(record) / 6)

    for i in range(result_length):
        label, score, xmin, ymin, xmax, ymax = record[0+(6*i)], record[1+(6*i)], record[2+(6*i)], record[3+(6*i)],record[4+(6*i)], record[5+(6*i)]
        cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0))
        cv2.putText(image, f"{label} {float(score) : .2f}%", (int(xmin), int(ymin)), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0))

    cv2.imwrite(f"/data/Datasets/SPC/full-name14/test/result_images/{file_name}.jpg", image)
    # cv2.imshow('result', image)
    # cv2.waitKey(0)           

def check_result(detect_results, annotations):
    final_total_result = []
    for det, annot in zip(detect_results, annotations):
        gt_bboxes, gt_labels = read_xml(annot, CLASSES)

        file_name = det.pop(0)
        pred_labels = []
        is_correct = False
        for i in range(int(len(det) / 6)):
            pred_label = det[0 + (6 * i)]
            pred_labels.append(pred_label)

        if len(gt_labels) == len(pred_labels) and sorted(gt_labels) == sorted(pred_labels):
            is_correct = True

        final_total_result.append([file_name, gt_labels, pred_labels, is_correct])
    return final_total_result
            
if __name__ == "__main__":
    model_path = "/data/Models/efficientdet_lite/full-name13-GAP6-300/full-name13-GAP6-300.tflite"
    images_path = "/data/Datasets/SPC/full-name14/test/images"
    label_path = "/data/Datasets/SPC/Labels/labels.txt"
    threshold = 0.7

    LABEL_FILE = pd.read_csv(label_path, sep=' ', index_col=False, header=None)
    CLASSES = LABEL_FILE[0].tolist()

    if not os.path.isdir("/data/Datasets/SPC/full-name14/test/result_images"):
        os.makedirs("/data/Datasets/SPC/full-name14/test/result_images")

    images = sorted(glob(f"{images_path}/*"))
    annotations = sorted(glob(f"/data/Datasets/SPC/full-name14/test/annotations/*"))

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
            result.extend(["No Results"])

        draw_result(result)
        total.append(result)
     
    final = check_result(total, annotations)
    # df = pd.DataFrame(total)
    # df.to_csv('/data/Datasets/SPC/full-name14/test/test-result.csv', index=False, header=None)

    df = pd.DataFrame(final)
    df.to_csv('/data/Datasets/SPC/full-name14/test/ttt.csv', index=False, header=["filename", "GT", "Pred", "GT == Pred"])