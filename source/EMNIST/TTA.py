import cv2, datetime, glob, string, os
import pandas as pd
import numpy as np
import tensorflow as tf
import albumentations as A

import matplotlib
matplotlib.use('Agg')
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


def get_images(test_files):
    images = []
    for n in tqdm(range(len(test_files))):
        image = test_files[n]
        image = cv2.imread(f'{test_image_path}/{image}.png')
        image2 = np.where((image <= 254) & (image != 0), 0, image)

        for _ in range(N_AUG):
            image2 = test_transforms(image=image2)['image']
            image2 = tf.keras.applications.resnet.preprocess_input(image2)
            images.append(image2)

    return np.array(images)


def get_predictions(test_images):
    models = sorted(glob.glob(f'{MODEL_PATH}/*/*.h5'))
    os.system('clear')
    print(f'NUM_MODELS : {len(models)}')

    final_preds = np.zeros((5000, 26), dtype=float)
    for idx, model in enumerate(models):
        model = tf.keras.models.load_model(model)

        start, end = 0, N_AUG
        tmp_preds = []
        while end <= len(test_images):
            multi_images = test_images[start:end]
            pred = model.predict(multi_images)
            pred = sum(pred) / N_AUG
            tmp_preds.append(pred)

            start += N_AUG
            end += N_AUG

        print(f'Model {idx+1}', np.array(tmp_preds).shape, len(tmp_preds))
        final_preds += tmp_preds

    final_preds /= len(models)
    final_preds = (final_preds > 0.5) * 1
    print(final_preds.shape)

    return final_preds
            

if __name__ == "__main__":
    N_AUG = 5
    IMG_SIZE = 256
    LOG_TIME = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    BASE_PATH = '/home/v100/tf_workspace/datasets/dirty_mnist_2'
    MODEL_PATH = '/home/v100/tf_workspace/model/dirty_mnist/v100_84_330_aug'
    
    test_image_path = f'{BASE_PATH}/test_dirty_mnist_2nd'
    sample_submission = f'{BASE_PATH}/sample_submission.csv'
    submission_df = pd.read_csv(sample_submission)
    
    CLASSES = list(string.ascii_lowercase)

    test_transforms = A.Compose([
        A.Resize(330, 330, p=1),
        A.OneOf([
            A.HorizontalFlip(p=0.8),
            A.VerticalFlip(p=0.8),
            A.RandomRotate90(p=0.9),
        ], p=1),
    ])

    test_images = submission_df['index']
    test_images = get_images(test_images)
    print(test_images.shape)

    predictions = get_predictions(test_images)

    result_df = pd.read_csv(sample_submission)
    result_df.iloc[:, 1:] = predictions
    result_df.to_csv(f'{BASE_PATH}/result_csv/result_{LOG_TIME}.csv', index=False)