import onnx, os
from onnx_tf.backend import prepare
import pandas as pd
import numpy as np
import tensorflow as tf
from tqdm import tqdm

def read_data_list(file_path, is_train):
    files = pd.read_csv(file_path, sep=' ', index_col=False, header=None)
    files = sorted(files[0].tolist())

    total_points = np.zeros(shape=[0, NUM_POINT, 3])
    total_labels = np.zeros(shape=[0, 1])

    for i in tqdm(range(len(files))):
        label = files[i].split('_')
        label = '_'.join(label[:-1])
        path = f'{DATA_PATH}/{label}/{files[i]}.txt'
        point = pd.read_csv(path, sep=',', index_col=False, header=None, names=['x', 'y', 'z', 'r', 'g', 'b'])

        point = point.loc[:,['x','y','z']]
        point = np.array(point)
        point = point[0:NUM_POINT, :]

        label = np.array([CLASSES.index(label)])

        total_points = np.append(total_points, [point], axis=0)
        total_labels = np.append(total_labels, [label], axis=0)
        # print(total_points.shape, total_labels.shape)

    return total_points, total_labels


DATA_PATH = '/data/datasets/modelnet40_normal_resampled'
CLASS_FILE = f'{DATA_PATH}/modelnet40_shape_names.txt'
CLASS_FILE = pd.read_csv(CLASS_FILE, sep=' ', index_col=False, header=None)
CLASSES = sorted(CLASS_FILE[0].tolist())
print(CLASSES)
NUM_POINT = 1024

VALID_FILE = f'{DATA_PATH}/modelnet40_test.txt'
test_points, test_labels = read_data_list(VALID_FILE, False)

onnx_model = onnx.load('onnx/convert_model.onnx')
# tf_rep = prepare(onnx_model, logging_level="WARN", auto_cast=True)
tf_rep = prepare(onnx_model)
tf_rep.export_graph('onnx/pointcutmix')

base_model = tf.keras.models.load_model('onnx/pointcutmix')
# print(base_model)
# print('loaded model inputs = ', base_model.signatures['serving_default'].inputs)
# print('loaded model outputs = ', base_model.signatures['serving_default'].outputs)

is_correct = 0
for i in tqdm(range(len(test_labels))):
    test_point, test_label = test_points[i], test_labels[i]
    test_point = np.transpose(test_point, axes=(1, 0))
    test_point = np.expand_dims(test_point, axis=0)
    test_point = test_point.astype('float32')
    
    pred = base_model(**{'x.1': test_point})
    pred = pred[0]
    pred = np.array(pred[0])
    
    idx = np.argmax(pred)
    name = CLASSES[idx]
    score = pred[idx]

    if name == CLASSES[int(test_label[0])]:
        is_correct += 1

print(f'Total Accuracy = {(is_correct / len(test_labels)) * 100 : .2f}')