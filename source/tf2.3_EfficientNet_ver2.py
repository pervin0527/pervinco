import os, cv2, datetime, pathlib, random, argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tqdm import tqdm
tqdm.pandas()

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


# Load data & Set hyper-parameters
AUTO = tf.data.experimental.AUTOTUNE
SAMPLE_LEN = 100
EPOCHS = 1000
BATCH_SIZE = 32 * strategy.num_replicas_in_sync
IMG_SIZE = 224


def preprocess_image(images, label=None):
    image = tf.io.read_file(images)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.keras.applications.efficientnet.preprocess_input(image)

    if label is None:
        return image

    else:
        return image, label


def make_tf_dataset(ds_path, is_train):
    ds_path = pathlib.Path(ds_path)

    images = list(ds_path.glob('*/*'))
    images = [str(path) for path in images]
    total_images = len(images)

    if is_train:
        random.shuffle(images)

    labels = sorted(item.name for item in ds_path.glob('*/') if item.is_dir())
    num_labels = len(labels)
    labels = dict((name, index) for index, name in enumerate(labels))
    labels = [labels[pathlib.Path(path).parent.name] for path in images]
    labels = tf.keras.utils.to_categorical(labels, num_classes=num_labels, dtype='float32')

    if is_train:
        dataset = (tf.data.Dataset
                   .from_tensor_slices((images, labels))
                   .map(preprocess_image, num_parallel_calls=AUTO)
                   .repeat()
                   .shuffle(512)
                   .batch(BATCH_SIZE)
                   .prefetch(AUTO)
        )
    
    else:
        dataset = (tf.data.Dataset
                   .from_tensor_slices((images, labels))
                   .map(preprocess_image, num_parallel_calls=AUTO)
                   .repeat()
                   .batch(BATCH_SIZE)
                   .prefetch(AUTO)
        )

    return dataset, total_images, num_labels


def tf_data_visualize(dataset):
    for image, label in dataset.take(1):
        print("Image shape: ", image.numpy().shape)
        print("Label: ", label.numpy().shape)

    image_batch, label_batch = next(iter(dataset))

    plt.figure(figsize=(10, 10))
    for i in range(16):
        ax = plt.subplot(4, 4, i + 1)
        plt.imshow((image_batch[i].numpy()).astype('uint8'))
        label = label_batch[i]
        plt.axis('off')

    plt.show()


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


def display_training_curves(history):
    acc = history.history['categorical_accuracy']
    val_acc = history.history['val_categorical_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(len(history.history['loss']))

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    
    plt.savefig(f'/{SAVED_PATH}/{LOG_TIME}/train_result.png')
    plt.show()


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image classification model Training")
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--visualize', type=str2bool, default=False)
    args = parser.parse_args()

    DATASET_NAME = args.dataset_name
    TRAIN_PATH = f'/data/backup/pervinco_2020/Auged_datasets/{DATASET_NAME}/train'
    VALID_PATH = f'/data/backup/pervinco_2020/Auged_datasets/{DATASET_NAME}/valid'

    train_dataset, train_total, train_labels = make_tf_dataset(TRAIN_PATH, True)
    valid_dataset, valid_total, valid_labels = make_tf_dataset(VALID_PATH, False)

    print(train_labels, valid_labels)

    if args.visualize == True:
        for i in range(3):
            tf_data_visualize(train_dataset)
            tf_data_visualize(valid_dataset)
    
    # Learning Rate Scheduler setup
    lrfn = build_lrfn()
    lr_schedule = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=1)

    # Checkpoint callback setup
    SAVED_PATH = f'/data/backup/pervinco_2020/model/{DATASET_NAME}'
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

    with strategy.scope():
        model = tf.keras.Sequential([tf.keras.applications.EfficientNetB0(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                                                                         weights='imagenet',
                                                                         include_top=False),

                                    tf.keras.layers.GlobalAveragePooling2D(),
                                    tf.keras.layers.Dense(train_labels, activation='softmax')])

    model.compile(optimizer='adam',
                    loss = 'categorical_crossentropy',
                    metrics = ['categorical_accuracy'])
    model.summary()

    TRAIN_STEPS_PER_EPOCH = int(tf.math.ceil(train_total/ BATCH_SIZE).numpy())
    VALID_STEP_PER_EPOCH = int(tf.math.ceil(valid_total / BATCH_SIZE).numpy())
    
    history = model.fit(train_dataset,
                        epochs=EPOCHS,
                        callbacks=[lr_schedule, checkpointer, earlystopper],
                        steps_per_epoch=TRAIN_STEPS_PER_EPOCH,
                        verbose=1,
                        validation_data=valid_dataset,
                        validation_steps=VALID_STEP_PER_EPOCH)

    model.save(f'{SAVED_PATH}/{LOG_TIME}/{DATASET_NAME}.h5')

    display_training_curves(history)