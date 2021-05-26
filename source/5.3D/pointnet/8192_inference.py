import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from matplotlib import pyplot as plt

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


def show_point_cloud(test_images, test_labels, results):
    fig = plt.figure(figsize=(15, 10))

    for i in range(len(results)):
        ax = fig.add_subplot(5, 2, i + 1, projection="3d")
        ax.scatter(test_images[i, :, 0], test_images[i, :, 1], test_images[i, :, 2])
        ax.set_title(f"P : {results[i]}, A : {CLASSES[int(test_labels[i][0])]}")
        ax.set_axis_off()

    plt.show()


def read_data_list(file_path):
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
        point = point[0:NUM_POINT, :]
        # print(point.shape)

        label = np.array([CLASSES.index(label)])
        # print(point.shape, label.shape)

        points = np.append(points, [point], axis=0)
        labels = np.append(labels, [label], axis=0)

    print(points.shape, labels.shape)

    return points, labels


if __name__ == "__main__":
    DATA_PATH = '/data/datasets/modelnet40_normal_resampled'
    BATCH_SIZE = 64
    NUM_POINT = 1024

    CLASS_FILE = f'{DATA_PATH}/modelnet40_shape_names.txt'
    CLASS_FILE = pd.read_csv(CLASS_FILE, sep=' ', index_col=False, header=None)
    CLASSES = sorted(CLASS_FILE[0].tolist())
    print(CLASSES)

    TEST_FILE = f'{DATA_PATH}/modelnet40_test.txt'
    test_points, test_labels = read_data_list(TEST_FILE)

    model = tf.keras.models.load_model('/data/Models/pointnet/2021.05.26_15:10/pointnet')
    model.summary()

    start, end = 0, 10
    test_image = test_points[start:end]
    test_label = test_labels[start:end]
    print(test_image.shape, test_label.shape)

    predictions = model.predict(test_image)
    print(predictions.shape)

    results = []
    for pred in predictions:
        idx = np.argmax(pred)
        label = CLASSES[idx]
        score = pred[idx]
        score = format(score, ".2f")

        results.append([label, score])

    show_point_cloud(test_image, test_label, results)

    predictions = model.predict(test_points)
    is_correct = 0

    for pred, answer in zip(predictions, test_labels):
        idx = np.argmax(pred)
        label = CLASSES[idx]

        if label == CLASSES[int(answer[0])]:
            is_correct += 1

    print(f'Total Accuracy = {(is_correct / len(test_labels)) * 100 : .2f}')