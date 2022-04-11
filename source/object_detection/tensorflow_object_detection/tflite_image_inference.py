import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import tflite_runtime.interpreter as tflite
from glob import glob
from tqdm import tqdm
from lxml.etree import Element, SubElement, tostring
from tensorflow.keras.preprocessing.image import img_to_array

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

    cv2.imwrite(f"{testset}/Records/{model_name}/{threshold}_result_img/{file_name}.jpg", image)
    # cv2.imshow('result', image)
    # cv2.waitKey(0)           

def check_result(detect_results, annotations):
    final_total_result = []
    for det, annot in zip(detect_results, annotations):
        gt_bboxes, gt_labels = read_xml(annot, CLASSES)

        file_name = det.pop(0)
        pred_labels = []
        pred_scores = []
        is_correct = False
        for i in range(int(len(det) / 6)):
            pred_label = det[0 + (6 * i)]
            pred_labels.append(pred_label)
            pred_score = det[1 + (6 * i)]
            pred_scores.append(pred_score)

        if len(gt_labels) == len(pred_labels) and sorted(gt_labels) == sorted(pred_labels):
            is_correct = True

        final_total_result.append([file_name, gt_labels, pred_labels, is_correct, pred_scores])
    return final_total_result

def VizGradCAM(model, image, number, interpolant=0.5, plot_results=True):
    original_img = np.asarray(image, dtype = np.float32)
    img = np.expand_dims(original_img, axis=0)
    prediction = model.predict(img)
    prediction_idx = np.argmax(prediction)

    last_conv_layer = next(x for x in model.layers[::-1] if isinstance(x, tf.keras.layers.Conv2D))
    target_layer = model.get_layer(last_conv_layer.name)

    with tf.GradientTape() as tape:
        gradient_model = tf.keras.Model([model.inputs], [target_layer.output, model.output])
        conv2d_out, prediction = gradient_model(img)
        loss = prediction[:, prediction_idx]

    gradients = tape.gradient(loss, conv2d_out)
    output = conv2d_out[0]
    weights = tf.reduce_mean(gradients[0], axis=(0, 1))
    activation_map = np.zeros(output.shape[0:2], dtype=np.float32)
    for idx, weight in enumerate(weights):
        activation_map += weight * output[:, :, idx]
    activation_map = cv2.resize(activation_map.numpy(), 
                                (original_img.shape[1], 
                                 original_img.shape[0]))
    activation_map = np.maximum(activation_map, 0)
    activation_map = (activation_map - activation_map.min()) / (activation_map.max() - activation_map.min())
    activation_map = np.uint8(255 * activation_map)


    heatmap = cv2.applyColorMap(activation_map, cv2.COLORMAP_JET)

    #superimpose heatmap onto image
    original_img = np.uint8((original_img - original_img.min()) / (original_img.max() - original_img.min()) * 255)
    cvt_heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    cvt_heatmap = img_to_array(cvt_heatmap)

    plt.imshow(np.uint8(original_img * interpolant + cvt_heatmap * (1 - interpolant)))
    plt.savefig(f"{testset}/Records/{model_name}/{threshold}_cam/{number:>05}.jpg")
            
if __name__ == "__main__":
    model_file = "/data/Models/efficientdet_lite/SPC-sample-set1-300/SPC-sample-set1-300.tflite"
    ckpt_file = "/data/Models/efficientdet_lite/SPC-sample-set1-300/ckpt"
    testset = "/data/Datasets/SPC/Testset/Normal"
    threshold = 0.45
    
    project_name, folder_name, model_name = testset.split('/')[-3], testset.split('/')[-1], model_file.split('/')[-1].split('.')[0]
    label_path = f"/data/Datasets/{project_name}/Labels/labels.txt"
    csv = f"{testset}/Records/{model_name}/{threshold}_result.csv"

    LABEL_FILE = pd.read_csv(label_path, sep=' ', index_col=False, header=None)
    CLASSES = LABEL_FILE[0].tolist()
    ckpt = tf.keras.models.load_model(ckpt_file)

    if not os.path.isdir(f"{testset}/Records/{model_name}"):
        os.makedirs(f"{testset}/Records/{model_name}/{threshold}_result_img")
        os.makedirs(f"{testset}/Records/{model_name}/{threshold}_cam")

    images = sorted(glob(f"{testset}/images/*"))
    annotations = sorted(glob(f"{testset}/annotations/*"))
    print(len(images), len(annotations))

    interpreter = tflite.Interpreter(model_path=model_file)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0].get('shape')
    input_width, input_height = input_shape[1], input_shape[2]
    input_dtype = input_details[0].get('dtype')

    total = []
    for idx in tqdm(range(len(images))):
        image_file = images[idx]
        image = cv2.imread(image_file)
        image = cv2.resize(image, (input_height, input_width))

        VizGradCAM(ckpt, image, idx)
        input_tensor = np.expand_dims(image, axis=0)
                
        interpreter.set_tensor(input_details[0]['index'], input_tensor.astype(np.uint8))
        # interpreter.set_tensor(input_details[0]['index'], input_tensor.numpy().astype(np.float32))
        interpreter.invoke()

        bboxes = interpreter.get_tensor(output_details[0]['index'])
        classes = interpreter.get_tensor(output_details[1]['index'])
        scores = interpreter.get_tensor(output_details[2]['index'])
        num_detections = interpreter.get_tensor(output_details[3]['index'])

        # scores = interpreter.get_tensor(output_details[0]['index'])
        # bboxes = interpreter.get_tensor(output_details[1]['index'])
        # num_detections = interpreter.get_tensor(output_details[2]['index'])
        # classes = interpreter.get_tensor(output_details[3]['index'])

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
    df.to_csv(csv, index=False, header=["filename", "GT", "Pred", "GT == Pred", "Top-1 Score"])