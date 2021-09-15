import cv2
import glob
import pandas as pd
import numpy as np
import tensorflow as tf

def convert_coordinates(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh

    return (x,y,w,h)

if __name__ == "__main__":
    model_file_path = "/data/Models/efficientdet_lite/efdet_d1_etri_augmentation.tflite"
    # model_file_path = "/data/Models/ssd_mobilenet_v2_etri/lite/custom.tflite"
    label_file_path = "/data/Datasets/Seeds/ETRI_detection/custom/labels.txt"
    testset_path = "/data/Datasets/testset/etri_detect_gt"
    output_path = "/home/barcelona/mAP/input/detection-results"
    threshold = 0.4

    LABEL_FILE = pd.read_csv(label_file_path, sep=' ', index_col=False, header=None)
    CLASSES = LABEL_FILE[0].tolist()

    interpreter = tf.lite.Interpreter(model_path=model_file_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0].get('shape')
    input_width, input_height = input_shape[1], input_shape[2]
        
    test_images = sorted(glob.glob(f'{testset_path}/*.jpg'))
    for test_image in test_images:
        filename = (test_image.split('/')[-1]).split('.')[0]
        image = tf.io.read_file(test_image)
        image = tf.image.decode_jpeg(image)
        image = tf.image.resize(image, (input_width, input_height))
        input_tensor = tf.expand_dims(image, axis=0)

        interpreter.set_tensor(input_details[0]['index'], input_tensor.numpy().astype(np.uint8))
        # interpreter.set_tensor(input_details[0]['index'], input_tensor.numpy().astype(np.float32))
        interpreter.invoke()

        boxes = interpreter.get_tensor(output_details[0]['index'])
        classes = interpreter.get_tensor(output_details[1]['index'])
        scores = interpreter.get_tensor(output_details[2]['index'])
        num_detections = interpreter.get_tensor(output_details[3]['index'])

        boxes = boxes[0]
        classes = classes[0]
        scores = scores[0]

        final_result = []
        print(filename)
        with open(f'{output_path}/{filename}.txt', 'w') as f:
            for index, score in enumerate(scores):
                    if score > threshold:
                        box = boxes[index]
                        label = CLASSES[int(classes[index])]
                        ymin, xmin, ymax, xmax = int(box[0] * input_width), int(box[1] * input_width), int(box[2] * input_width), int(box[3] * input_width)
                        # print(xmin, ymin, ymax, xmax)
                        result = convert_coordinates((input_width, input_height), (float(xmin), float(xmax), float(ymin), float(ymax)))
                        f.write(str(label) + " " + str(score) + " " + " ".join([("%.6f" % a) for a in result]) + '\n')
        f.close()