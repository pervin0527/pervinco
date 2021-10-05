import cv2
import numpy as np
import pandas as pd
import onnxruntime as ort

def load_model(path):
    print(f"Model Path : {path}")
    ort_session = ort.InferenceSession(path)

    input_shape = ort_session.get_inputs()[0].shape
    print(f"Input Shape : {input_shape}")

    return ort_session, input_shape


def pre_processing(path, input_shape):
    original_image = cv2.imread(path)
    # original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    if input_shape[1] != 3:
        test_image = cv2.resize(original_image, (input_shape[1], input_shape[2]))
        test_image = np.expand_dims(test_image, axis=0)

        print(f"Channel Last : {test_image.shape}")
        return test_image, original_image, input_shape[1], input_shape[2]


    else:
        test_image = cv2.resize(original_image, (input_shape[2], input_shape[3]))
        test_image = np.transpose(test_image, [2,0,1])
        test_image = np.expand_dims(test_image, axis=0)

        print(f"Channel First : {test_image.shape}")
        return test_image, original_image, input_shape[2], input_shape[3]


def post_processing(detection_result, image, threshold, width, height):
    image = cv2.resize(image, (width, height))
    result = []

    bboxes = detection_result[0][0]
    classes = detection_result[1][0]
    scores = detection_result[2][0]

    for idx in range(len(scores)):
        if scores[idx] > threshold:
            result.append((int(classes[idx]), scores[idx], bboxes[idx]))
            ymin, xmin, ymax, xmax = bboxes[idx][0], bboxes[idx][1], bboxes[idx][2], bboxes[idx][3]
            xmin *= int(width)
            ymin *= int(height)
            xmax *= int(width)
            ymax *= int(height)

            cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 255, 0))
            cv2.putText(image, f"{label[int(classes[idx]-1)]} {scores[idx]:.2f}%", (int(xmin), int(ymin)), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0))

    print(result)
    cv2.imshow('result', image)
    cv2.waitKey(0)


def inference(ort_session, test_x):
    ort_inputs = {ort_session.get_inputs()[0].name: test_x.astype(np.uint8)}
    ort_outs = ort_session.run(None, ort_inputs)
    # print(ort_outs)

    return ort_outs


if __name__ == "__main__":
    model_path = "/data/Models/efficientdet_lite/model.onnx"
    image_path = "/data/Datasets/testset/ETRI_cropped_large/test_sample_08.jpg"
    label_path = "/data/Datasets/Seeds/ETRI_detection/labels/labels.txt"
    detection_threshold = 0.7

    label = pd.read_csv(label_path, sep=' ', index_col=False, header=None)    
    label = label[0].tolist()

    ort_session, input_tensor_shape = load_model(model_path)
    test_x, image, width, height = pre_processing(image_path, input_tensor_shape)

    detection_result = inference(ort_session, test_x)
    post_processing(detection_result, image, detection_threshold, width, height)
    