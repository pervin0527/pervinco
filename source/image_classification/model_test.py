import os, cv2, argparse, time, pathlib
import numpy as np
import tensorflow as tf
import pandas as pd

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


def read_images(path):
    test_path = pathlib.Path(path)
    test_path = list(test_path.glob('*.jpg'))
    test_images = sorted([str(path) for path in test_path])

    return test_images


def preprocess_images(images):
    images = tf.io.read_file(images)
    images = tf.image.decode_jpeg(images, channels=3)
    images = tf.image.resize(images, (IMG_SIZE, IMG_SIZE))
    images = tf.keras.applications.efficientnet.preprocess_input(images)

    return images


def get_test_dataset(images):
    testset = tf.data.Dataset.from_tensor_slices(images)
    testset = testset.map(preprocess_images, num_parallel_calls=AUTOTUNE)
    testset = testset.batch(BATCH_SIZE)
    testset = testset.prefetch(AUTOTUNE)

    return testset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="testset classification")
    parser.add_argument('--testset_path', type=str)
    parser.add_argument('--model_path', type=str)
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model_path)
    test_images = read_images(args.testset_path)
    label_path = args.model_path.split('/')[:-1]
    label_path = '/'.join(label_path)
    label_file = pd.read_csv(f'{label_path}/main_labels.txt', sep=' ', index_col=False, header=None)

    CLASSES = sorted(label_file[0].tolist())
    IMG_SIZE = 224
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    BATCH_SIZE = len(test_images)

    testset = get_test_dataset(test_images)
    
    predictions = model.predict(testset)
    os.system('clear')
    print(len(test_images))
    
    for pred, file_name in zip(predictions, test_images):
        pred = [(idx, score) for idx, score in enumerate(pred)]
        
        score_board = []
        for _ in range(3):
            tmp = max(pred, key=lambda x : x[1])
            pred.pop(tmp[0])

            label = CLASSES[tmp[0]]
            score = format(tmp[1], ".2f")
            score_board.append((label, score))

        print(file_name, score_board)
