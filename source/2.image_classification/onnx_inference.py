import tensorflow as tf
import efficientnet.tfkeras as efn
import matplotlib.pyplot as plt
import cv2, glob
import numpy as np
import onnxruntime
import pandas as pd

model_path = '/home/ubuntu/pervinco/model/natural_images/2021.05.06_17:19/main_model.onnx'

label_path = model_path.split('/')[:-1]
label_path = '/'.join(label_path)
label_file = pd.read_csv(f'{label_path}/main_labels.txt', sep=' ', index_col=False, header=None)
classes = sorted(label_file[0].tolist())
print(f'CLASSES : {classes}')

dataset_name = model_path.split('/')[-3]
test_images = sorted(glob.glob(f'/home/ubuntu/pervinco/datasets/test_images/{dataset_name}/*.jpg'))
print(f'TOTAL TEST IMAGES : {len(test_images)}')

total_images = []
for image in test_images:
    image = cv2.imread(image)
    image = cv2.resize(image, (224, 224))
    total_images.append(image)

total_images = np.array(total_images, dtype=np.float32)
total_images = [total_images]

sess_options = onnxruntime.SessionOptions()
sess = onnxruntime.InferenceSession(model_path, sess_options)

# data = [x.astype(np.float32)]
# input_names = sess.get_inputs()
# feed = zip(sorted(i_.name for i_ in input_names), data)

input_names = sess.get_inputs()
input_datas = zip(sorted(i_.name for i_ in input_names), total_images)

prediction = sess.run(None, dict(input_datas))
for pred, file_name in zip(prediction[0], test_images):
    idx = np.argmax(pred)
    class_name = classes[idx]
    score = pred[idx]

    file_name = file_name.split('/')[-1]

    print(f"FILE_NAME : {file_name}, CLASS_NAME : {class_name}, SCORE : {score}")

