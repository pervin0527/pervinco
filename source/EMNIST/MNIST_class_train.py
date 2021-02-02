import cv2, datetime, os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
import albumentations as A
from tqdm import tqdm
from functools import partial
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


def data_preprocess(images, labels):
    images = tf.image.grayscale_to_rgb(images)
    images = tf.cast(images, tf.float32) / 255.0
    images = tf.image.resize(images, (IMG_SIZE, IMG_SIZE))
    
    labels = tf.one_hot(labels, len(CLASSES), dtype='float32')

    return images, labels


def get_dataset(images, labels):
    images = tf.data.Dataset.from_tensor_slices(images)
    labels = tf.data.Dataset.from_tensor_slices(labels)

    dataset = tf.data.Dataset.zip((images, labels))
    dataset = dataset.map(data_preprocess, num_parallel_calls=AUTOTUNE)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTOTUNE)

    return dataset


def get_model():
    with strategy.scope():
        base_model = tf.keras.applications.EfficientNetB5(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                                                          weights="imagenet", # noisy-student
                                                          include_top=False)
        base_model.trainable = True
            
        avg = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
        output = tf.keras.layers.Dense(len(CLASSES), activation="softmax")(avg)
        model = tf.keras.Model(inputs=base_model.input, outputs=output)

    model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics = ['categorical_accuracy'])
    # model.summary()

    return model


def build_lrfn(lr_start=0.00001, lr_max=0.00005, 
               lr_min=0.00001, lr_rampup_epochs=5, 
               lr_sustain_epochs=0, lr_exp_decay=.8):
    lr_max = lr_max * strategy.num_replicas_in_sync

    def lrfn(epoch):
        if epoch < lr_rampup_epochs:
            lr = (lr_max - lr_start) / lr_rampup_epochs * epoch + lr_start
        elif epoch < lr_rampup_epochs + lr_sustain_epochs:
            lr = lr_max
        else:
            lr = (lr_max - lr_min) *\
                 lr_exp_decay**(epoch - lr_rampup_epochs\
                                - lr_sustain_epochs) + lr_min
        return lr
    return lrfn


def train_cross_validate(images, labels, folds=5):
    kfold = KFold(n_splits=5, random_state=0, shuffle=True)

    for f, (train_index, valid_index) in enumerate(kfold.split(images, labels)):
        print("##################################################################################################################")
        print('FOLD', f + 1)
        # print(train_index, valid_index)
        train_images, train_labels = images[train_index[0] : (train_index[-1] + 1)], labels[train_index[0] : (train_index[-1] + 1)]
        valid_images, valid_labels = images[valid_index[0] : (valid_index[-1] + 1)], labels[valid_index[0] : (valid_index[-1] + 1)]

        # print(train_images.shape, train_labels.shape)
        # print(valid_images.shape, valid_labels.shape)

        lrfn = build_lrfn()
        lr_schedule = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=1)

        checkpoint_path = f'/{SAVED_PATH}/{LOG_TIME}/{f+1}-{WEIGHT_FNAME}'
        checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                        monitor='val_categorical_accuracy',
                                                        save_best_only=True,
                                                        mode='max')
        earlystopper = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

        train_total, valid_total = len(train_images), len(valid_images)
        TRAIN_STEPS_PER_EPOCH = int(tf.math.ceil(train_total/ BATCH_SIZE).numpy())
        VALID_STEP_PER_EPOCH = int(tf.math.ceil(valid_total / BATCH_SIZE).numpy())

        model = get_model()
        
        if f == 0:
            model.summary()

        history = model.fit(get_dataset(train_images, train_labels),
                            epochs=EPOCHS,
                            callbacks=[lr_schedule, checkpointer, earlystopper],
                            steps_per_epoch=TRAIN_STEPS_PER_EPOCH,
                            verbose=1,
                            validation_data=get_dataset(valid_images, valid_labels),
                            validation_steps=VALID_STEP_PER_EPOCH)

        model.save(f'{SAVED_PATH}/{LOG_TIME}/{f+1}_model.h5')


if __name__ == "__main__":
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    IMG_SIZE = 32
    IMAGE_SIZE = [IMG_SIZE, IMG_SIZE]
    EPOCHS = 1000
    BATCH_SIZE = 128
    DATASET_NAME = 'mnist'

    SAVED_PATH = f'/data/tf_workspace/model/{DATASET_NAME}'
    LOG_TIME = datetime.datetime.now().strftime("%Y.%m.%d_%H:%M")
    WEIGHT_FNAME = '{epoch:02d}-{val_categorical_accuracy:.2f}.hdf5'

    if not(os.path.isdir(f'/{SAVED_PATH}/{LOG_TIME}')):
        os.makedirs(f'/{SAVED_PATH}/{LOG_TIME}')

    # Load MNIST/Letters dataset
    (images1, labels1), (images2, labels2) = tfds.as_numpy(tfds.load('emnist/letters', split=['train', 'test'], batch_size=-1, as_supervised=True,))

    # Merge train, valid
    total_images = np.concatenate((images1, images2))
    total_labels = np.concatenate((labels1, labels2))
    print(total_images.shape, total_labels.shape)

    total_labels -= 1
    CLASSES = np.unique(total_labels)
    print(CLASSES)

    train_cross_validate(total_images, total_labels)