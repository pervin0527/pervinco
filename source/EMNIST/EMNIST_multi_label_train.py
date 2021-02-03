import numpy as np
import cv2, datetime, os
import pandas as pd
import tensorflow as tf
import albumentations as A
from functools import partial
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


def read_dataset():
    CLASSES = [c for c in df]
    CLASSES = CLASSES[1:]

    images = []
    labels = []
    image_dir = '/home/v100/tf_workspace/datasets/dirty_mnist_2/dirty_mnist_2nd'
    for idx in tqdm(range(len(df))):
        file_name = str(df.iloc[idx, 0]).zfill(5)
        image = f'{image_dir}/{file_name}.png'
        label = df.iloc[idx, 1:].values.astype('float')

        images.append(image)
        labels.append(label)

    return images, labels, CLASSES


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
    images = tf.io.read_file(images)
    images = tf.image.decode_png(images, channels=3)
    images = tf.keras.applications.efficientnet.preprocess_input(images)
    
    return images, labels


def get_train_dataset(images, labels):
    images = tf.data.Dataset.from_tensor_slices(images)
    labels = tf.data.Dataset.from_tensor_slices(labels)

    dataset = tf.data.Dataset.zip((images, labels))
    dataset = dataset.map(partial(data_preprocess), num_parallel_calls=AUTOTUNE)
    dataset = dataset.repeat()
    dataset = dataset.map(partial(process_data), num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTOTUNE)

    return dataset


def get_valid_dataset(images, labels):
    images = tf.data.Dataset.from_tensor_slices(images)
    labels = tf.data.Dataset.from_tensor_slices(labels)

    dataset = tf.data.Dataset.zip((images, labels))
    dataset = dataset.map(data_preprocess, num_parallel_calls=AUTOTUNE)
    dataset = dataset.repeat()
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
        base_model = tf.keras.applications.EfficientNetB5(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                                                          weights="imagenet", # noisy-student
                                                          include_top=False)
        for layer in base_model.layers:
            layer.trainable = True
            
        avg = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
        output = tf.keras.layers.Dense(len(CLASSES), activation="sigmoid")(avg)
        model = tf.keras.Model(inputs=base_model.input, outputs=output)

    model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['binary_accuracy'])
    
    return model


def train_cross_validate(images, labels, folds=5):
    histories = []
    models = []

    kfold = KFold(folds, shuffle=True, random_state=777)
    for f, (train_index, valid_index) in enumerate(kfold.split(images, labels)):
        print('FOLD', f + 1)
        train_x, train_y = images[train_index[0] : (train_index[-1] + 1)], labels[train_index[0] : (train_index[-1] + 1)]
        valid_x, valid_y = images[valid_index[0] : (valid_index[-1] + 1)], labels[valid_index[0] : (valid_index[-1] + 1)]

        if not(os.path.isdir(f'{SAVED_PATH}/{LOG_TIME}')):
            os.makedirs(os.path.join(f'{SAVED_PATH}/{LOG_TIME}'))

        WEIGHT_FNAME = '{epoch:02d}-{val_binary_accuracy:.2f}.hdf5'
        checkpoint_path = f'{SAVED_PATH}/{LOG_TIME}/{f+1}-{WEIGHT_FNAME}'
        cb_checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                             monitor='val_binary_accuracy',
                                                             save_best_only=True,
                                                             mode='max')
        lrfn = build_lrfn()
        cb_lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose = True)        
        cb_early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)      

        TRAIN_STEPS_PER_EPOCH = int(tf.math.ceil(len(train_x) / BATCH_SIZE).numpy())
        VALID_STEPS_PER_EPOCH = int(tf.math.ceil(len(valid_x) / BATCH_SIZE).numpy())

        model = get_model()
        history = model.fit(get_train_dataset(train_x, train_y), 
                            steps_per_epoch = TRAIN_STEPS_PER_EPOCH,
                            epochs = EPOCHS,
                            validation_data = get_valid_dataset(valid_x, valid_y),
                            validation_steps = VALID_STEPS_PER_EPOCH,
                            verbose=1,
                            callbacks = [cb_lr_callback, cb_early_stopping, cb_checkpointer])

        model.save(f'{SAVED_PATH}/{LOG_TIME}/{f+1}_{DATASET_NAME}.h5')

        models.append(model)
        histories.append(history)

    return histories, models


if __name__ == "__main__":
    EPOCHS = 1000
    IMG_SIZE = 300
    IMAGE_SIZE = [IMG_SIZE, IMG_SIZE]
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    BATCH_SIZE = 64
    DATASET_NAME = 'dirty_mnist'
    SAVED_PATH = f'/home/v100/tf_workspace/model/{DATASET_NAME}'
    LOG_TIME = datetime.datetime.now().strftime("%Y.%m.%d_%H:%M")

    transforms = A.Compose([
                    A.MedianBlur(blur_limit=3, always_apply=True, p=1),
                    A.HorizontalFlip(p=0.4),
                    A.VerticalFlip(p=0.3),
                ])

    if not(os.path.isdir(f'/{SAVED_PATH}/{LOG_TIME}')):
        os.makedirs(f'/{SAVED_PATH}/{LOG_TIME}')

    df = pd.read_csv('/home/v100/tf_workspace/datasets/dirty_mnist_2/dirty_mnist_2nd_answer.csv')
    total_images, total_labels, CLASSES = read_dataset()
    histories, models = train_cross_validate(total_images, total_labels, folds=5)   

    # train_ds = get_training_dataset(total_images, total_labels)
    # for item in train_ds.take(1):
    #     print(item)