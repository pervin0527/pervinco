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


def preprocessing(x, y):
    x = tf.cast(x, dtype=tf.float32)
    y = tf.cast(y, dtype=tf.int64)
    y = tf.one_hot(y, depth=NUM_CLASSES)
    y = tf.squeeze(y, axis=0)
    
    return (x, y)


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

    TEST_FILES = '/data/datasets/modelnet40_ply_hdf5_2048/test_files.txt'
    TEST_FILES = get_data_files(TEST_FILES)

    model = tf.saved_model.load('/data/Models/pointnet/tf')

    test_data, test_answer = load_h5(TEST_FILES[0])
    test_data = test_data[:, 0:NUM_POINT, :]

    is_correct = 0
    for i in tqdm(range(len(test_answer))):
        test_point, test_label = test_data[i], test_answer[i]
        # test_point = np.reshape(test_point, (3, 1024))
        test_point = np.transpose(test_point, axes=(1, 0))
        test_point = np.expand_dims(test_point, axis=0)

        pred = model(**{'x.1' : test_point})
        pred = np.array(pred[0])
        
        idx = np.argmax(pred)
        name = CLASSES[idx]
        score = pred[0][idx]

        if name == CLASSES[test_label[0]]:
            is_correct += 1

    print(f'Total Accuracy = {(is_correct / len(test_answer)) * 100 : .2f}')