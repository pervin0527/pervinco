import cv2, pathlib, datetime, os
import numpy as np
import pandas as pd
import tensorflow as tf

from matplotlib import pyplot as plt
from functools import partial
from tqdm import tqdm
from sklearn.model_selection import KFold

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


# Disable AutoShard.
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF


def get_dataset(df, is_custom):
    CLASSES = [c for c in df]
    CLASSES = CLASSES[1:]

    # print(len(df))
    X = np.zeros([len(df), IMG_SIZE, IMG_SIZE, 3], dtype=np.uint8)
    y = np.zeros([len(df), len(CLASSES)], dtype=np.uint8)

    for idx in tqdm(range(len(df))):
        if is_custom:
            file_name = str(df.iloc[idx, 0])
            image = cv2.imread(f'{CUSTOM_DS_PATH}/{file_name}.png')

        else:
            file_name = str(df.iloc[idx, 0]).zfill(5)
            image = cv2.imread(f'{TRAIN_DS_PATH}/{file_name}.png')
            image = np.where((image <= 254) & (image != 0), 0, image)

        X[idx] = image
        label = df.iloc[idx, 1:].values.astype('float')
        y[idx] = label

    return X, y, CLASSES


def normalize_image(image, label):
    image = tf.image.resize(image, [RE_SIZE, RE_SIZE])
    image = tf.cast(image, tf.float32)
    image = tf.keras.applications.resnet.preprocess_input(image)

    label = tf.cast(label, tf.float32)
    
    return image, label


def make_tf_dataset(images, labels):
    images = tf.data.Dataset.from_tensor_slices(images)
    labels = tf.data.Dataset.from_tensor_slices(labels)

    dataset = tf.data.Dataset.zip((images, labels))
    dataset = dataset.repeat()
    dataset = dataset.map(normalize_image, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTOTUNE)
    dataset = dataset.with_options(options)

    return dataset


def get_model():
    with strategy.scope():
        base_model = tf.keras.applications.EfficientNetB0(input_shape=(RE_SIZE, RE_SIZE, 3),
                                                          weights='imagenet', # noisy-student
                                                          include_top=False)
        base_model.trainable = True
            
        avg = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
        output = tf.keras.layers.Dense(26, activation="sigmoid")(avg)
        model = tf.keras.Model(inputs=base_model.input, outputs=output)

    # model.compile(optimizer=gctf.optimizers.adam(), loss = 'binary_crossentropy', metrics = ['binary_accuracy'])
    model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['binary_accuracy'])
    
    return model


if __name__ == "__main__":
    IMG_SIZE = 256
    EPOCHS = 100
    BATCH_SIZE = 20
    RE_SIZE = 224
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    CUSTOM_DS_PATH = '/data/backup/pervinco/test_code/custom_mnist'
    TRAIN_DS_PATH = '/data/backup/pervinco/datasets/dirty_mnist_2/dirty_mnist_2nd'

    CUSTOM_CSV_PATH = '/data/backup/pervinco/test_code/custom_mnist.csv'
    CUSTOM_DF = pd.read_csv(CUSTOM_CSV_PATH)

    TRAIN_CSV_PATH = '/data/backup/pervinco/test_code/train-kfold-0.csv'
    TRAIN_DF = pd.read_csv(TRAIN_CSV_PATH)

    VALID_CSV_PATH = '/data/backup/pervinco/test_code/valid-kfold-0.csv'
    VALID_DF = pd.read_csv(VALID_CSV_PATH)

    custom_x, custom_y, _ = get_dataset(CUSTOM_DF, True)
    train_x, train_y, _ = get_dataset(TRAIN_DF, False)
    valid_x, valid_y, _ = get_dataset(VALID_DF, False)

    train_x = np.concatenate((train_x, custom_x), axis=0)
    train_y = np.concatenate((train_y, custom_y), axis=0)

    os.system('clear')
    print(train_x.shape, train_y.shape)
    print(valid_x.shape, valid_y.shape)

    cb_early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    TRAIN_STEPS_PER_EPOCH = int(tf.math.ceil(len(train_x) / BATCH_SIZE).numpy())
    VALID_STEPS_PER_EPOCH = int(tf.math.ceil(len(valid_x) / BATCH_SIZE).numpy())

    model = get_model()
    model.fit(make_tf_dataset(train_x, train_y), 
              steps_per_epoch = TRAIN_STEPS_PER_EPOCH,
              epochs = EPOCHS,
              validation_data = make_tf_dataset(valid_x, valid_y),
              validation_steps = VALID_STEPS_PER_EPOCH,
              verbose=1,
              callbacks = [cb_early_stopping])
