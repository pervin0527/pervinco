import cv2, os
import tensorflow as tf
import pandas as pd
import numpy as np
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"]="1"
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
        tf.config.experimental.set_virtual_device_configuration(gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=11000)])
  except RuntimeError as e:
    print(e)


def get_result(df):
    model = tf.keras.models.load_model('/data/tf_workspace/models/landmark_classification/k-fold/ensemble_landmark_cls.h5')

    df = df['id']
    start = 0
    end = 0
    step = int(len(df) / 4)

    result = []
    for _ in range(4):
        end += step
        images = []

        for file in df[start:end]:
            print(file)
            folder = file[0]
            image = cv2.imread(test_ds_path + '/' + folder + '/' + file + '.JPG')
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
            image = image / 255.0
            images.append(image)

        images = np.array(images)
        print(start, end, images.shape)
        start = end

        predictions = model.predict(images)
        for pred in predictions:
            landmark_id = np.argmax(pred)
            landmark_name = str(CLASSES[landmark_id])
            conf = pred[landmark_id]

            print({'landmark_id' : landmark_id, 'landmark_name' : landmark_name, 'conf' : conf})
            result.append({'landmark_id' : landmark_id, 'conf' : conf})

    return result


if __name__ == "__main__":
    IMG_HEIGHT = 270
    IMG_WIDTH = 480
    category = pd.read_csv('/data/tf_workspace/datasets/public/category.csv')
    CLASSES = category['landmark_name'].tolist()

    test_ds_path = '/data/tf_workspace/datasets/public/test/'
    submission = pd.read_csv('/data/tf_workspace/datasets/public/sample_submission.csv')

    result = get_result(submission)    
    result_df = pd.DataFrame(result)
    result_df.to_csv('/data/tf_workspace/datasets/public/results/fold2_result.csv')