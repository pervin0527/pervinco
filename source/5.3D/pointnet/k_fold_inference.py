import trimesh
import numpy as np
import tensorflow as tf
import pandas as pd
from train import get_data_files, load_h5
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


def read_data_list(file_path, start, end):
    files = pd.read_csv(file_path, sep=' ', index_col=False, header=None)
    files = sorted(files[0].tolist())

    points = np.zeros(shape=[0, NUM_POINT, 3])
    labels = np.zeros(shape=[0, 1])

    for i in tqdm(range(len(files))):
        label = files[i].split('_')
        label = label[:-1]
        label = '_'.join(label)
        path = f'{DATA_PATH}/{label}/{files[i]}.txt'
        point = pd.read_csv(path, sep=',', index_col=False, header=None, names=['x', 'y', 'z', 'r', 'g', 'b'])

        point = point.loc[:,['x','y','z']]
        point = np.array(point)
        point = point[start:end, :]
        # print(point.shape)

        label = np.array([CLASSES.index(label)])
        # print(point.shape, label.shape)

        points = np.append(points, [point], axis=0)
        labels = np.append(labels, [label], axis=0)

    print(points.shape, labels.shape)

    return points, labels


def show_point_cloud(test_images, test_labels, results):
    fig = plt.figure(figsize=(15, 10))

    for i in range(len(results)):
        ax = fig.add_subplot(5, 2, i + 1, projection="3d")
        ax.scatter(test_images[i, :, 0], test_images[i, :, 1], test_images[i, :, 2])
        ax.set_title(f"P : {results[i]}, A : {CLASSES[test_labels[i][0]]}")
        ax.set_axis_off()

    plt.show()


if __name__ == "__main__":
    NUM_POINT = 1024
    CLASSES = pd.read_csv('/data/datasets/modelnet40_ply_hdf5_2048/shape_names.txt',
                          sep=' ',
                          index_col=False,
                          header=None)
    CLASSES = sorted(CLASSES[0].tolist())

    DATA_PATH = '/data/datasets/modelnet40_normal_resampled'
    TEST_FILE = f'{DATA_PATH}/modelnet40_test.txt'

    start = 0
    end = NUM_POINT
    final_predictions = np.zeros(shape=(2468, 40))

    for idx in range(5):
        model = tf.keras.models.load_model(f'/data/Models/pointnet/2021.05.27_11:56/pointnet_{idx}')
        test_points, test_labels = read_data_list(TEST_FILE, start, end)

        predictions = model.predict(test_points)
        final_predictions += predictions

        start += NUM_POINT
        end += NUM_POINT

    print(final_predictions.shape)
    final_predictions /= 5

    is_correct = 0
    for pred, answer in zip(final_predictions, test_labels):
        idx = np.argmax(pred)
        label = CLASSES[idx]

        if label == CLASSES[int(answer[0])]:
            is_correct += 1

    print(f'Total Accuracy = {(is_correct / len(test_labels)) * 100 : .2f}')