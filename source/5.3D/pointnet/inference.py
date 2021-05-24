import trimesh
import numpy as np
import tensorflow as tf
from train import get_data_files, load_h5
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
    NUM_POINT = 2048
    CLASSES = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf',
               'bottle', 'bowl', 'car', 'chair', 'cone',
               'cup', 'curtain', 'desk', 'door', 'dresser',
               'flower_pot', 'glass_box', 'guitar', 'keyboard', 'lamp',
               'laptop', 'mantel', 'monitor', 'night_stand', 'person',
               'piano', 'plant', 'radio', 'range_hood', 'sink',
               'sofa', 'stairs', 'stool', 'table', 'tent',
               'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']
    CLASSES = sorted(CLASSES)

    TEST_FILES = '/data/datasets/modelnet40_ply_hdf5_2048/test_files.txt'
    TEST_FILES = get_data_files(TEST_FILES)

    test_data, test_answer = load_h5(TEST_FILES[0])
    test_data = test_data[:, 0:NUM_POINT, :]

    # print(test_data.shape, test_label.shape)

    start = 0
    # start = np.random.randint((len(test_label) - 10))
    end = start + 10
    print(start, end)
    
    test_image = test_data[start:end]
    test_label = test_answer[start:end]
    print(test_image.shape, test_label.shape)

    model = tf.keras.models.load_model('/data/Models/pointnet/2021.05.24_11:40/pointnet')
    model.summary()

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

    predictions = model.predict(test_data)
    is_correct = 0

    for pred, answer in zip(predictions, test_answer):
        idx = np.argmax(pred)
        label = CLASSES[idx]

        if label == CLASSES[answer[0]]:
            is_correct += 1

    print(f'Total Accuracy = {(is_correct / len(test_answer)) * 100 : .2f}')