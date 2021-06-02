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
        point = point[0:1024, :]

        label = np.array([CLASSES.index(label)])

        total_points = np.append(total_points, [point], axis=0)
        total_labels = np.append(total_labels, [label], axis=0)

        return total_points, total_labels


DATA_PATH = '/data/datasets/modelnet40_normal_resampled'
CLASS_FILE = f'{DATA_PATH}/modelnet40_shape_names.txt'
CLASS_FILE = pd.read_csv(CLASS_FILE, sep=' ', index_col=False, header=None)
CLASSES = sorted(CLASS_FILE[0].tolist())
print(CLASSES)
NUM_POINT = 1024

TRAIN_FILE = f'{DATA_PATH}/modelnet40_train.txt'
VALID_FILE = f'{DATA_PATH}/modelnet40_test.txt'

onnx_model = onnx.load('onnx/convert_model.onnx')
# tf_rep = prepare(onnx_model, logging_level="WARN", auto_cast=True)
tf_rep = prepare(onnx_model)
tf_rep.export_graph('onnx/pointcutmix')

base_model = tf.keras.models.load_model('onnx/pointcutmix')
# print(base_model)
# print('loaded model inputs = ', base_model.signatures['serving_default'].inputs)
# print('loaded model outputs = ', base_model.signatures['serving_default'].outputs)

input_tensor = tf.random.uniform(shape=[1, 3, 1024])
pred = base_model(**{'x.1': input_tensor})
print(pred)