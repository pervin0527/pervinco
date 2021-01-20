import os, datetime, argparse, pathlib, random, cv2
import numpy as np
import tensorflow as tf
import albumentations as A

from matplotlib import pyplot as plt
from functools import partial
from sklearn.model_selection import train_test_split

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
EPOCHS = 1000
BATCH_SIZE = 16 * strategy.num_replicas_in_sync
IMG_SIZE = 224

transforms = A.Compose([
                        A.Resize(IMG_SIZE, IMG_SIZE, 3, p=1),
                        A.HorizontalFlip(p=0.4),
                        A.VerticalFlip(p=0.3),
                        A.Blur(p=0.1),
    
                        A.OneOf([
                            A.RandomContrast(p=0.5, limit=(-0.5, 0.3)),
                            A.RandomBrightness(p=0.5, limit=(-0.2, 0.3))
                        ], p=0.5)
                ])


def basic_processing(ds_path):
    ds_path = pathlib.Path(ds_path)

    images = list(ds_path.glob('*/*'))
    images = [str(path) for path in images]

    labels = sorted(item.name for item in ds_path.glob('*/') if item.is_dir())
    labels_len = len(labels)
    labels = dict((name, index) for index, name in enumerate(labels))
    labels = [labels[pathlib.Path(path).parent.name] for path in images]
    labels = tf.keras.utils.to_categorical(labels, num_classes=labels_len, dtype='float32')

    return images, labels, labels_len


def aug_fn(image):
    data = {"image":image}
    aug_data = transforms(**data)
    aug_img = aug_data["image"]
    aug_img = tf.cast(aug_img, tf.float32)

    return aug_img


def process_data(image, label):
    aug_img = tf.numpy_function(func=aug_fn, inp=[image], Tout=tf.float32)

    return aug_img, label


def preprocess_train_image(images):
    image = tf.io.read_file(images)
    image = tf.image.decode_jpeg(image, channels=3)
    # image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.keras.applications.efficientnet.preprocess_input(image)

    return image


def preprocess_valid_image(images):
    image = tf.io.read_file(images)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.keras.applications.efficientnet.preprocess_input(image)

    return image


def get_train_dataset(images, labels):
    images = tf.data.Dataset.from_tensor_slices(images)
    images = images.map(preprocess_train_image, num_parallel_calls=AUTO)
    labels = tf.data.Dataset.from_tensor_slices(labels)

    dataset = tf.data.Dataset.zip((images, labels))
    dataset = dataset.repeat()
    dataset = dataset.map(partial(process_data), num_parallel_calls=AUTO)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO)

    return dataset


def get_valid_dataset(images, labels):
    images = tf.data.Dataset.from_tensor_slices(images)
    images = images.map(preprocess_valid_image, num_parallel_calls=AUTO)
    labels = tf.data.Dataset.from_tensor_slices(labels)
    
    dataset = tf.data.Dataset.zip((images, labels))
    dataset = dataset.repeat()
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO)

    return dataset


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


def tf_data_visualize(augmentation_element):
    row, col, idx = 5, 4, 0
    row = min(row, BATCH_SIZE // col)

    for (image, label) in augmentation_element:
        print(image.shape, label.shape)
        image = image / 255.0
        plt.figure(figsize=(15, int(15 * row / col)))
        for j in range(row * col):
            plt.subplot(row, col, j + 1)
            plt.axis('off')
            plt.imshow(image[j, ])

        plt.savefig(f'{SAVED_PATH}/{LOG_TIME}/result_{idx}.jpg')
        plt.show()
        idx += 1

        if idx == 3:
            break


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_model():
    with strategy.scope():
        base_model = tf.keras.applications.EfficientNetB1(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                                                          weights="imagenet", # noisy-student
                                                          include_top=False)
        for layer in base_model.layers:
            layer.trainable = True

        avg = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
        batch_norm = tf.keras.layers.BatchNormalization()(avg)
        drop_out = tf.keras.layers.Dropout(0.2)(batch_norm)
        output = tf.keras.layers.Dense(num_classes, activation="softmax")(drop_out)
        
        model = tf.keras.Model(inputs=base_model.input, outputs=output)

    model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics = ['categorical_accuracy'])
    model.summary()
    return model


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image classification model Training")
    parser.add_argument('--input_dataset', type=str)
    parser.add_argument('--visualize', type=str2bool, default=False)
    args = parser.parse_args()

    dataset = args.input_dataset
    DATASET_NAME = dataset.split('/')[-1]

    total_images, total_labels, num_classes = basic_processing(dataset)
    train_images, valid_images, train_labels, valid_labels = train_test_split(total_images, total_labels, test_size=.3, random_state=777)
    train_dataset = get_train_dataset(train_images, train_labels)
    valid_dataset = get_valid_dataset(valid_images, valid_labels)

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

    if args.visualize:
        tf_data_visualize(train_dataset)
        tf_data_visualize(valid_dataset)

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
    history = model.fit(train_dataset,
                        epochs=EPOCHS,
                        callbacks=[lr_schedule, checkpointer, earlystopper],
                        steps_per_epoch=TRAIN_STEPS_PER_EPOCH,
                        verbose=1,
                        validation_data=valid_dataset,
                        validation_steps=VALID_STEP_PER_EPOCH)

    model.save(f'{SAVED_PATH}/{LOG_TIME}/{DATASET_NAME}.h5')

    display_training_curves(history)