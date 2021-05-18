import numpy as np
import tensorflow as tf
from train import get_data_files, load_h5

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

    TEST_FILES = '/data/data/modelnet40_ply_hdf5_2048/test_files.txt'
    TEST_FILES = get_data_files(TEST_FILES)

    test_data, test_label = load_h5(TEST_FILES[0])
    test_data = test_data[:, 0:NUM_POINT, :]
    
    test_image = test_data[:5]
    answers = test_label[:5]
    print(test_image.shape, answers.shape)

    model = tf.keras.models.load_model('./model/pointnet')
    model.summary()

    predictions = model.predict(test_image)
    print(predictions.shape)

    for pred, answer in zip(predictions, answers):
        idx = np.argmax(pred)
        label = CLASSES[idx]
        score = pred[idx]

        answer = CLASSES[answer[0]]

        print(f"predict result : {label}, {score}, answer : {answer}")