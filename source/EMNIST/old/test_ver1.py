import cv2, string
import pandas as pd
import numpy as np
import datetime
import tensorflow as tf

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


def get_images(test_files):
    images = []
    for path in test_files:
        image = cv2.imread(f'{test_image_path}/{path}.png')
        image2 = np.where((image <= 254) & (image != 0), 0, image)
        image3 = cv2.dilate(image2, kernel=np.ones((2, 2), np.uint8), iterations=1)
        image4 = cv2.medianBlur(image3, 5)
        image5 = image4 - image2
        image5 = tf.keras.applications.resnet.preprocess_input(image5)
        images.append(image5)

    return np.array(images)


def load_and_predict(test_images):
    model = tf.keras.models.load_model(MODEL_PATH)

    pred = model.predict(test_images)
    pred = (pred > 0.6) * 1

    return pred


if __name__ == "__main__":
    IMG_SIZE = 256
    LOG_TIME = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    BASE_PATH = '/data/backup/pervinco/datasets/dirty_mnist_2'
    MODEL_PATH = '/data/backup/pervinco/model/dirty_mnist/2021_02_16_09_12/dirty_mnist.h5'
    
    test_image_path = f'{BASE_PATH}/test_dirty_mnist_2nd'
    sample_submission = f'{BASE_PATH}/sample_submission.csv'
    submission_df = pd.read_csv(sample_submission)
    
    CLASSES = list(string.ascii_lowercase)

    test_images = submission_df['index']
    test_images = get_images(test_images)
    print(test_images.shape)

    predictions = load_and_predict(test_images)

    result_df = pd.read_csv(sample_submission)
    result_df.iloc[:, 1:] = predictions
    result_df.to_csv(f'{BASE_PATH}/result_csv/result_{LOG_TIME}.csv', index=False)