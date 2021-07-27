# python3 -m tf2onnx.convert --tflite fire_efdet_d0.tflite --opset 9 --output test.onnx --dequantize --inputs-as-nchw serving_default_images:0
# https://github.com/microsoft/Windows-Machine-Learning/issues/386
# https://www.google.com/search?q=failed+to+load+model+with+error%3A+unknown+model+file+format+version.&oq=&aqs=chrome.0.69i59i450l8.201823287j0j15&sourceid=chrome&ie=UTF-8

import cv2
import onnxruntime as ort
import numpy as np
import pandas as pd
# np.set_printoptions(formatter={'float_kind': lambda x: "{0:0.3f}".format(x)})

IMG_PATH = "/data/Datasets/testset/ETRI_cropped_large/test_sample_24.jpg"
MODEL_PATH = "/home/barcelona/test/ssd_mb_v2/test.onnx"
LABEL_FILE = pd.read_csv('/data/Datasets/Seeds/ETRI_detection/labels.txt', sep=' ', index_col=False, header=None)
CLASSES = sorted(LABEL_FILE[0].tolist())
THRESH_HOLD = 0.4
IMG_SIZE = 320

ort_session = ort.InferenceSession(MODEL_PATH)

image = cv2.imread(IMG_PATH)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
# test_x = np.transpose(image, [2, 0, 1])
test_x = np.expand_dims(image, axis=0)
print(test_x.shape)

ort_inputs = {ort_session.get_inputs()[0].name: test_x.astype(np.float32)}
ort_outs = ort_session.run(None, ort_inputs)

"""EfficientDet"""
# print(ort_outs[0].shape) # Bounding Box
# print(ort_outs[1].shape) # categories of the detected boxes
# print(ort_outs[2].shape) # scores of the detected boxes
# print(ort_outs[3].shape) # number of the detected boxes

# bboxes = ort_outs[0][0]
# labels = ort_outs[1][0]
# scores = ort_outs[2][0]

# final_result = []
# for idx in range(len(scores)):
#     if scores[idx] > THRESH_HOLD:
#         final_result.append((int(labels[idx]), scores[idx], bboxes[idx]))
#         ymin, xmin, ymax, xmax = bboxes[idx][0], bboxes[idx][1], bboxes[idx][2], bboxes[idx][3]
#         xmin *= IMG_SIZE
#         ymin *= IMG_SIZE
#         xmax *= IMG_SIZE
#         ymax *= IMG_SIZE

#         cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0))

"""MobileNet"""
print(ort_outs[0].shape) # detection anchor indices
print(ort_outs[1].shape) # detection boxes
print(ort_outs[2].shape) # detection classes
print(ort_outs[3].shape) # detection multiclass scores
print(ort_outs[4].shape) # detection scores

bboxes = ort_outs[1][0]
labels = ort_outs[2][0]
scores = ort_outs[4][0]

final_result = []
for idx in range(len(scores)):
    if scores[idx] > THRESH_HOLD:
        final_result.append((int(labels[idx]), scores[idx], bboxes[idx]))
        ymin, xmin, ymax, xmax = bboxes[idx][0], bboxes[idx][1], bboxes[idx][2], bboxes[idx][3]
        xmin *= IMG_SIZE
        ymin *= IMG_SIZE
        xmax *= IMG_SIZE
        ymax *= IMG_SIZE

        cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0))

""" RESULT """
print(final_result)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
cv2.imshow('result', image)
cv2.waitKey(0)