import cv2, datetime, os
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from sklearn.model_selection import KFold

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


def get_dataset():
    CLASSES = [c for c in df]
    CLASSES = CLASSES[1:]

    images = []
    labels = []
    image_dir = '/data/tf_workspace/datasets/data_2/dirty_mnist'
    for idx in tqdm(range(len(df))):
        file_name = str(df.iloc[idx, 0]).zfill(5)
        image = f'{image_dir}/{file_name}.png'
        label = df.iloc[idx, 1:].values.astype('float')

        images.append(image)
        labels.append(label)

    return images, labels, CLASSES


def preprocess_image(images):
    images = tf.io.read_file(images)
    images = tf.image.decode_png(images, channels=3)
    images = tf.keras.applications.efficientnet.preprocess_input(images)
    # images = tf.cast(images, tf.float32) / 255.0
    images = tf.reshape(images, [*IMAGE_SIZE, 3])

    return images


def get_training_dataset(images, labels):
    images = tf.data.Dataset.from_tensor_slices(images)
    images = images.map(preprocess_image, num_parallel_calls=AUTOTUNE)
    labels = tf.data.Dataset.from_tensor_slices(labels)

    dataset = (tf.data.Dataset
               .zip((images, labels))
               .repeat()
               .batch(BATCH_SIZE)
               .prefetch(AUTOTUNE)
    )

    return dataset


def get_validation_dataset(images, labels, do_onehot=True):
    images = tf.data.Dataset.from_tensor_slices(images)
    images = images.map(preprocess_image, num_parallel_calls=AUTOTUNE)
    labels = tf.data.Dataset.from_tensor_slices(labels)
    
    dataset = tf.data.Dataset.zip((images, labels))
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTOTUNE)

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


def get_model():
    with strategy.scope():
        base_model = tf.keras.applications.EfficientNetB7(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
                                                          weights='imagenet',
                                                          include_top=False)
        base_model.trainable = True
        model = tf.keras.Sequential([base_model,
                                     tf.keras.layers.GlobalAveragePooling2D(),
                                     tf.keras.layers.Dense(len(CLASSES), activation='sigmoid')])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    return model


def train_cross_validate(images, labels, folds=5):
    histories = []
    models = []

    kfold = KFold(folds, shuffle=True, random_state=777)
    for f, (train_index, valid_index) in enumerate(kfold.split(images, labels)):
        print('FOLD', f + 1)
        train_x, train_y = images[train_index[0] : (train_index[-1] + 1)], labels[train_index[0] : (train_index[-1] + 1)]
        valid_x, valid_y = images[valid_index[0] : (valid_index[-1] + 1)], labels[valid_index[0] : (valid_index[-1] + 1)]
        STEPS_PER_EPOCH = int(tf.math.ceil(len(images) / BATCH_SIZE).numpy())

        if not(os.path.isdir(f'{SAVED_PATH}/{LOG_TIME}')):
            os.makedirs(os.path.join(f'{SAVED_PATH}/{LOG_TIME}'))

        WEIGHT_FNAME = '{epoch:02d}-{val_accuracy:.2f}.hdf5'
        checkpoint_path = f'{SAVED_PATH}/{LOG_TIME}/{f+1}-{WEIGHT_FNAME}'
        cb_checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                             monitor='val_accuracy',
                                                             save_best_only=True,
                                                             mode='max')
        lrfn = build_lrfn()
        cb_lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose = True)        
        cb_early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)      

        model = get_model()
        history = model.fit(get_training_dataset(train_x, train_y), 
                            steps_per_epoch = STEPS_PER_EPOCH,
                            epochs = EPOCHS,
                            callbacks = [cb_lr_callback, cb_early_stopping, cb_checkpointer],
                            validation_data = get_validation_dataset(valid_x, valid_y),
                            verbose=1)

        model.save(f'{SAVED_PATH}/{LOG_TIME}/{f+1}_mnist.h5')

        models.append(model)
        histories.append(history)

    return histories, models


if __name__ == "__main__":
    EPOCHS = 1000
    IMAGE_SIZE = [256, 256]
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    BATCH_SIZE = 5 * strategy.num_replicas_in_sync
    SAVED_PATH = f'/data/tf_workspace/model/mnist'
    LOG_TIME = datetime.datetime.now().strftime("%Y.%m.%d_%H:%M")

    if not(os.path.isdir(f'/{SAVED_PATH}/{LOG_TIME}')):
        os.makedirs(f'/{SAVED_PATH}/{LOG_TIME}')

    df = pd.read_csv('/data/tf_workspace/datasets/data_2/dirty_mnist_answer.csv')
    total_images, total_labels, CLASSES = get_dataset()
    # histories, models = train_cross_validate(total_images, total_labels, folds=5)   

    train_ds = get_training_dataset(total_images, total_labels)
    for item in train_ds.take(1):
        print(item)