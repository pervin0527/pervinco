import cv2, datetime, os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
import albumentations as A
from tqdm import tqdm
from functools import partial

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


def aug_fn(image):
    data = {"image":image}
    aug_data = transforms(**data)
    aug_img = aug_data["image"]
    aug_img = tf.cast(aug_img, tf.float32)

    return aug_img


def process_data(image, label):
    aug_img = tf.numpy_function(func=aug_fn, inp=[image], Tout=tf.float32)

    return aug_img, label


def data_preprocess(images, labels):
    images = tf.image.resize(images, (IMG_SIZE, IMG_SIZE))
    images = tf.image.grayscale_to_rgb(images)
    images = tf.keras.applications.efficientnet.preprocess_input(images)
    labels = tf.one_hot(labels, CLASSES)

    return images, labels


def get_train_dataset(images, labels):
    images = tf.data.Dataset.from_tensor_slices(images)
    labels = tf.data.Dataset.from_tensor_slices(labels)

    dataset = tf.data.Dataset.zip((images, labels))
    dataset = dataset.map(data_preprocess, num_parallel_calls=AUTOTUNE)
    dataset = dataset.map(partial(process_data), num_parallel_calls=AUTOTUNE)
    dataset = dataset.repeat()
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.shuffle(512)
    dataset = dataset.prefetch(AUTOTUNE)

    return dataset


def get_valid_dataset(images, labels):
    images = tf.data.Dataset.from_tensor_slices(images)
    labels = tf.data.Dataset.from_tensor_slices(labels)

    dataset = tf.data.Dataset.zip((images, labels))
    dataset = dataset.map(data_preprocess, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTOTUNE)

    return dataset


def get_model():
    with strategy.scope():
        base_model = tf.keras.applications.EfficientNetB5(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                                                          weights="imagenet", # noisy-student
                                                          include_top=False)
        base_model.trainable = False
        x = base_model.output
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(1024, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        predictions = tf.keras.layers.Dense(CLASSES, activation='softmax')(x)
        
        model = tf.keras.Model(inputs=base_model.input, outputs=predictions)

    # optimizer = tf.keras.optimizers.RMSprop(lr=0.0001, decay=1e-6)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.summary()

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



if __name__ == "__main__":
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    IMG_SIZE = 56
    IMAGE_SIZE = [IMG_SIZE, IMG_SIZE]
    EPOCHS = 1000
    CLASSES = 62
    BATCH_SIZE = 32 * strategy.num_replicas_in_sync
    DATASET_NAME = 'mnist'

    transforms = A.Compose([A.RandomRotate90(p=1),])


    (train_images, train_labels), (valid_images, valid_labels) = tfds.as_numpy(tfds.load('emnist/byclass',
                                                                                      split=['train', 'test'],
                                                                                      batch_size=-1,
                                                                                      as_supervised=True,
                                                                                    #   shuffle_files=True,
                                                                                      ))
                                                        
    print(train_images.shape, train_labels.shape)
    print(valid_images.shape, valid_labels.shape)

    lrfn = build_lrfn()
    lr_schedule = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=1)

    SAVED_PATH = f'/data/tf_workspace/model/{DATASET_NAME}'
    LOG_TIME = datetime.datetime.now().strftime("%Y.%m.%d_%H:%M")
    WEIGHT_FNAME = '{epoch:02d}-{val_categorical_accuracy:.2f}.hdf5'
    checkpoint_path = f'/{SAVED_PATH}/{LOG_TIME}/{WEIGHT_FNAME}'

    if not(os.path.isdir(f'/{SAVED_PATH}/{LOG_TIME}')):
        os.makedirs(f'/{SAVED_PATH}/{LOG_TIME}')

    checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                      monitor='val_categorical_accuracy',
                                                      save_best_only=True,
                                                      mode='max')
    earlystopper = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

    train_total, valid_total = len(train_images), len(valid_images)
    print(train_total, valid_total)
    TRAIN_STEPS_PER_EPOCH = int(tf.math.ceil(train_total/ BATCH_SIZE).numpy())
    VALID_STEP_PER_EPOCH = int(tf.math.ceil(valid_total / BATCH_SIZE).numpy())

    model = get_model()    
    history = model.fit(get_train_dataset(train_images, train_labels),
                        epochs=EPOCHS,
                        callbacks=[lr_schedule, checkpointer, earlystopper],
                        steps_per_epoch=TRAIN_STEPS_PER_EPOCH,
                        verbose=1,
                        validation_data=get_valid_dataset(valid_images, valid_labels),
                        validation_steps=VALID_STEP_PER_EPOCH)

    model.save(f'{SAVED_PATH}/{LOG_TIME}/main_model.h5')
    model.save(f'{SAVED_PATH}/{LOG_TIME}/pb_model', save_format='tf')