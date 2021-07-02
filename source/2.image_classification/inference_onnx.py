import cv2
import onnx
import onnxruntime as ort
import numpy as np
import pandas as pd

MODEL_PATH = "/data/Models/ETRI_custom/2021.07.02_14:47/converted.onnx"
OUTPUT_PATH = MODEL_PATH.split('/')[:-1]
OUTPUT_PATH = '/'.join(OUTPUT_PATH)

LABEL_FILE = pd.read_csv(f'{OUTPUT_PATH}/main_labels.txt', sep=' ', index_col=False, header=None)
CLASSES = LABEL_FILE[0].tolist()

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

ort_session = ort.InferenceSession(f"{OUTPUT_PATH}/converted_mod.onnx")

test_img = "/data/Datasets/Augmentations/ETRI_custom/test_sample5.jpg"
test_img = cv2.imread(test_img)
test_img = cv2.resize(test_img, (224, 224))
X_test = np.expand_dims(test_img, axis=0)
X_test = np.reshape(X_test, [1, 3, 224, 224])
print(X_test.shape)

ort_inputs = {ort_session.get_inputs()[0].name: X_test.astype(np.float32)}
ort_outs = ort_session.run(None, ort_inputs)
img_out_y = ort_outs[0]

idx = np.argmax(img_out_y)
print(CLASSES[idx])