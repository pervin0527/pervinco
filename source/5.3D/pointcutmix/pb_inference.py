import trimesh, h5py
import numpy as np
import tensorflow as tf
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

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


def show_point_cloud(test_images, test_labels, results):
    fig = plt.figure(figsize=(15, 10))

    for i in range(len(results)):
        ax = fig.add_subplot(5, 2, i + 1, projection="3d")
        ax.scatter(test_images[i, :, 0], test_images[i, :, 1], test_images[i, :, 2])
        ax.set_title(f"P : {results[i]}, A : {CLASSES[int(test_labels[i][0])]}")
        ax.set_axis_off()

    plt.show()


if __name__ == "__main__":
    NUM_POINT = 1024
    DATA_PATH = '/data/datasets/modelnet40_normal_resampled'
    CLASS_FILE = f'{DATA_PATH}/modelnet40_shape_names.txt'
    CLASSES = pd.read_csv(CLASS_FILE, sep=' ', index_col=False, header=None)
    CLASSES = sorted(CLASSES[0].tolist())

    TEST_FILES = f'{DATA_PATH}/modelnet40_test.txt'
    test_points, test_labels = read_data_list(TEST_FILES, False)

    model = tf.saved_model.load('/data/Models/pointcutmix')

    is_correct = 0
    results = []
    for i in tqdm(range(len(test_labels))):
        test_point, test_label = test_points[i], test_labels[i]
        # test_point = np.reshape(test_point, (3, 1024))
        test_point = np.transpose(test_point, axes=(1, 0))
        test_point = np.expand_dims(test_point, axis=0)
        test_point = test_point.astype('float32')

        pred = model(**{'x.1' : test_point})
        pred = pred[0]
        pred = np.array(pred[0])
        
        idx = np.argmax(pred)
        name = CLASSES[idx]
        score = pred[idx]
        score = format(score, ".2f")

        if i < 10:
            results.append([name, score])

        if name == CLASSES[int(test_label[0])]:
            is_correct += 1

    print(f'Total Accuracy = {(is_correct / len(test_labels)) * 100 : .2f}')
    
    show_point_cloud(test_points[:10], test_labels[:10], results)