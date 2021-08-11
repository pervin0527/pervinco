# python3 -m tf2onnx.convert --tflite fire_efdet_d0.tflite --opset 9 --output test.onnx --dequantize --inputs-as-nchw serving_default_images:0
# https://github.com/microsoft/Windows-Machine-Learning/issues/386
# https://www.google.com/search?q=failed+to+load+model+with+error%3A+unknown+model+file+format+version.&oq=&aqs=chrome.0.69i59i450l8.201823287j0j15&sourceid=chrome&ie=UTF-8

import argparse
import cv2
import onnxruntime as ort
import numpy as np
import pandas as pd

def load_model(path):
    print(path)
    ort_session = ort.InferenceSession(path)

    return ort_session

def image_process(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (int(args.image_size), int(args.image_size)))

    if args.image_format == "channel_last":
        image = np.expand_dims(image, axis=0)

    elif args.image_format == "channel_first":
        image = np.transpose(image, [2, 0, 1])
        image = np.expand_dims(image, axis=0)

    return image

def inference(session, test_x):
    if args.model_name == "efficientdet_lite":
        ort_inputs = {ort_session.get_inputs()[0].name: test_x.astype(np.uint8)}

    else:
        ort_inputs = {ort_session.get_inputs()[0].name: test_x.astype(np.float32)}

    ort_outs = ort_session.run(None, ort_inputs)
    return ort_outs

def pred_post_process(predictions, image_path):
    final_result = []
    image = cv2.imread(image_path)
    image = cv2.resize(image, (int(args.image_size), int(args.image_size)))
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if args.model_name == "efficientdet_lite":
        bboxes = predictions[0][0]
        labels = predictions[1][0]
        scores = predictions[2][0]

        # print(bboxes.shape, labels.shape, scores.shape)

        for idx in range(len(scores)):
            if scores[idx] > float(args.threshold):
                final_result.append((int(labels[idx]), scores[idx], bboxes[idx]))
                ymin, xmin, ymax, xmax = bboxes[idx][0], bboxes[idx][1], bboxes[idx][2], bboxes[idx][3]
                xmin *= int(args.image_size)
                ymin *= int(args.image_size)
                xmax *= int(args.image_size)
                ymax *= int(args.image_size)

                cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0))

    else:
        bboxes = predictions[1][0]
        labels = predictions[2][0]
        scores = predictions[4][0]

        for idx in range(len(scores)):
            if scores[idx] > float(args.threshold):
                final_result.append((int(labels[idx]), scores[idx], bboxes[idx]))
                ymin, xmin, ymax, xmax = bboxes[idx][0], bboxes[idx][1], bboxes[idx][2], bboxes[idx][3]
                xmin *= int(args.image_size)
                ymin *= int(args.image_size)
                xmax *= int(args.image_size)
                ymax *= int(args.image_size)

                cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0))

    print(CLASSES)
    print(final_result)

    for result in final_result:
        print(f"CLASSES : {CLASSES[result[0]]}")

    cv2.imshow('result', image)
    cv2.waitKey(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inferencing onnx format model")
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--image_path", type=str)
    parser.add_argument("--image_size", type=str)
    parser.add_argument("--image_format", type=str)
    parser.add_argument("--label_file_path", type=str)
    parser.add_argument("--threshold", type=str)
    args = parser.parse_args()

    LABEL_FILE = pd.read_csv(args.label_file_path, sep=' ', index_col=False, header=None)
    CLASSES = LABEL_FILE[0].tolist()

    ort_session = load_model(args.model_path)
    test_image = image_process(args.image_path)

    result = inference(ort_session, test_image)
    # print(result)

    pred_post_process(result, args.image_path)