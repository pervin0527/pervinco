import os
import cv2
import pathlib
import onnx
import onnxruntime as ort
import numpy as np
import pandas as pd
import tensorflow as tf

IMG_SIZE = 224
MODEL_PATH = "/data/Models/ETRI_cropped_large/2021.07.07_15:58/converted.onnx"
OUTPUT_PATH = MODEL_PATH.split('/')[:-1]
OUTPUT_PATH = '/'.join(OUTPUT_PATH)

DATASET_NAME = MODEL_PATH.split('/')[3]
LABEL_FILE = pd.read_csv(f'{OUTPUT_PATH}/main_labels.txt', sep=' ', index_col=False, header=None)
CLASSES = sorted(LABEL_FILE[0].tolist())

onnx_model = onnx.load(MODEL_PATH)

endpoint_names = ['input_2:0', 'Identity']

for i in range(len(onnx_model.graph.node)):
	for j in range(len(onnx_model.graph.node[i].input)):
		if onnx_model.graph.node[i].input[j] in endpoint_names:
			print('-'*60)
			print(onnx_model.graph.node[i].name)
			print(onnx_model.graph.node[i].input)
			print(onnx_model.graph.node[i].output)

			onnx_model.graph.node[i].input[j] = onnx_model.graph.node[i].input[j].split(':')[0]

	for j in range(len(onnx_model.graph.node[i].output)):
		if onnx_model.graph.node[i].output[j] in endpoint_names:
			print('-'*60)
			print(onnx_model.graph.node[i].name)
			print(onnx_model.graph.node[i].input)
			print(onnx_model.graph.node[i].output)

			onnx_model.graph.node[i].output[j] = onnx_model.graph.node[i].output[j].split(':')[0]

for i in range(len(onnx_model.graph.input)):
	if onnx_model.graph.input[i].name in endpoint_names:
		print('-'*60)
		print(onnx_model.graph.input[i])
		onnx_model.graph.input[i].name = onnx_model.graph.input[i].name.split(':')[0]

for i in range(len(onnx_model.graph.output)):
	if onnx_model.graph.output[i].name in endpoint_names:
		print('-'*60)
		print(onnx_model.graph.output[i])
		onnx_model.graph.output[i].name = onnx_model.graph.output[i].name.split(':')[0]

onnx.save(onnx_model, f'{OUTPUT_PATH}/converted_mod.onnx')

print(CLASSES)
ort_session = ort.InferenceSession(f"{OUTPUT_PATH}/converted_mod.onnx")

## Preprocessing Channel first
test_path = f"/data/Datasets/testset/{DATASET_NAME}"
test_path = pathlib.Path(test_path)
test_images = list(test_path.glob('*.jpg'))
test_images = sorted([str(path) for path in test_images])

# os.system('clear')
for test_img in test_images:
	file_name = test_img.split('/')[-1]
	image = cv2.imread(test_img)
	
	# X_test = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	# X_test = cv2.resize(X_test, (IMG_SIZE, IMG_SIZE))

	X_test = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

	X_test = np.transpose(X_test, [2, 0, 1])
	X_test = np.expand_dims(X_test, axis=0)
	# print(X_test.shape)

	ort_inputs = {ort_session.get_inputs()[0].name: X_test.astype(np.float32)}
	ort_outs = ort_session.run(None, ort_inputs)
	img_out_y = ort_outs[0]

	idx = np.argmax(img_out_y)
	score = img_out_y[0][idx]
	score = format(score, ".2f")
	# print(file_name, CLASSES[idx], score)

	image = cv2.resize(image, (640, 480))
	cv2.putText(image, f"{CLASSES[idx]} : {score}%", (0, 40), cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=(0, 255, 0), thickness=2)	
	cv2.imshow("result", image)
	cv2.waitKey(0)