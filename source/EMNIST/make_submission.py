import cv2, string
import pandas as pd
import numpy as np
import datetime
import tensorflow as tf

# GPU setup
gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 1:
    try:
        print("ActivateMulti GPU")
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

def get_model():
    with strategy.scope():
        base_model = tf.keras.applications.EfficientNetB2(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                                                          weights='imagenet', # noisy-student
                                                          include_top=False)
        for layer in base_model.layers:
            layer.trainable = True
            
        avg = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
        output = tf.keras.layers.Dense(len(CLASSES), activation="sigmoid")(avg)
        model = tf.keras.Model(inputs=base_model.input, outputs=output)

    model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['binary_accuracy'])
    
    return model

def get_images(test_files):
    images = []
    for path in test_files:
        image = cv2.imread(f'{test_image_path}/{path}.png')
        image2 = np.where((image <= 254) & (image != 0), 0, image)
        image3 = cv2.dilate(image2, kernel=np.ones((2, 2), np.uint8), iterations=1)
        image4 = cv2.medianBlur(image3, 5)
        image5 = image4 - image2
        # image5 = np.expand_dims(image5, axis=0)
        
        images.append(image5)

    return np.array(images)

def load_and_predict(test_images):
    model = get_model()
    model.load_weights(MODEL_PATH)

    pred = model.predict(test_images)
    pred = (pred > 0.5) * 1

    return pred

if __name__ == "__main__":
    IMG_SIZE = 256
    LOG_TIME = datetime.datetime.now().strftime("%Y.%m.%d_%H:%M:%s")
    BASE_PATH = '/data/backup/pervinco/dataset/dirty_mnist/dirty_mnist_2'
    MODEL_PATH = '/data/backup/pervinco/model/dirty_mnist/2021.02.09_14:10/1-244-1.00.hdf5'
    
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