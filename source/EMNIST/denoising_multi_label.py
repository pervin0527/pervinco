import cv2, pathlib, datetime, os
import numpy as np
import pandas as pd
import tensorflow as tf
import albumentations as A
import matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot as plt
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


def get_dataset():
    CLASSES = [c for c in df]
    CLASSES = CLASSES[1:]

    # print(len(df))
    X = np.zeros([len(df), IMG_SIZE, IMG_SIZE, 3], dtype=np.uint8)
    y = np.zeros([len(df), len(CLASSES)], dtype=np.uint8)

    for idx in tqdm(range(len(df))):
        file_name = str(df.iloc[idx, 0]).zfill(5)
        image = cv2.imread(f'{TRAIN_DS_PATH}/{file_name}.png')

        image2 = np.where((image <= 254) & (image != 0), 0, image)
        image3 = cv2.dilate(image2, kernel=np.ones((2, 2), np.uint8), iterations=1)
        image4 = cv2.medianBlur(image3, 5)
        image5 = image4 - image2

        X[idx] = image5

        label = df.iloc[idx, 1:].values.astype('float')
        y[idx] = label

    return X, y, CLASSES


def aug_fn(image):
    data = {"image":image}
    aug_data = transforms(**data)
    aug_img = aug_data["image"]
    aug_img = tf.cast(aug_img / 255.0, tf.float32)

    return aug_img


def process_data(image, label):
    aug_img = tf.numpy_function(func=aug_fn, inp=[image], Tout=tf.float32)

    return aug_img, label


def get_train_dataset(images, labels):
    images = tf.data.Dataset.from_tensor_slices(images)
    labels = tf.data.Dataset.from_tensor_slices(labels)

    dataset = tf.data.Dataset.zip((images, labels))
    dataset = dataset.repeat()
    dataset = dataset.map(partial(process_data), num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTOTUNE)

    return dataset


def get_valid_dataset(images, labels):
    images = tf.data.Dataset.from_tensor_slices(images)
    labels = tf.data.Dataset.from_tensor_slices(labels)

    dataset = tf.data.Dataset.zip((images, labels))
    dataset = dataset.repeat()
    dataset = dataset.map(partial(process_data), num_parallel_calls=AUTOTUNE)
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


def tf_data_visualize(augmentation_element, name):
    row, col, idx = 5, 4, 0
    row = min(row, BATCH_SIZE // col)

    for (image, label) in augmentation_element:
        print(image.shape, label.shape)
        
        # image = tf.cast(image, tf.uint8) * 255

        plt.figure(figsize=(15, int(15 * row / col)))
        for j in range(row * col):
            plt.subplot(row, col, j + 1)
            plt.axis('off')
            plt.imshow(image[j, ])

        plt.savefig(f'{SAVED_PATH}/{LOG_TIME}/{name}_{idx}.jpg')
        plt.show()
        idx += 1

        if idx == 3:
            break


def train_cross_validate(images, labels, folds=5):
    histories = []
    models = []

    kfold = KFold(folds, shuffle=True, random_state=777)
    for f, (train_index, valid_index) in enumerate(kfold.split(images, labels)):
        print('FOLD', f + 1)
        train_x, train_y = images[train_index[0] : (train_index[-1] + 1)], labels[train_index[0] : (train_index[-1] + 1)]
        valid_x, valid_y = images[valid_index[0] : (valid_index[-1] + 1)], labels[valid_index[0] : (valid_index[-1] + 1)]

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

        tf_data_visualize(get_train_dataset(train_x, train_y), 'train')
        tf_data_visualize(get_valid_dataset(valid_x, valid_y), 'valid')

        model = get_model()
        history = model.fit(get_train_dataset(train_x, train_y), 
                            steps_per_epoch = TRAIN_STEPS_PER_EPOCH,
                            epochs = EPOCHS,
                            validation_data = get_valid_dataset(valid_x, valid_y),
                            validation_steps = VALID_STEPS_PER_EPOCH,
                            verbose=1,
                            callbacks = [cb_lr_callback, cb_early_stopping, cb_checkpointer],
                            # callbacks = [cb_checkpointer],
                            )

        model.save(f'{SAVED_PATH}/{LOG_TIME}/{f+1}_dmnist.h5')

        models.append(model)
        histories.append(history)

    return histories, models


if __name__ == "__main__":
    EPOCHS = 100
    IMG_SIZE = 256
    IMAGE_SIZE = [IMG_SIZE, IMG_SIZE]
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    BATCH_SIZE = 10 * strategy.num_replicas_in_sync
    DS_PATH = '/data/tf_workspace/datasets/dirty_mnist_2'
    SAVED_PATH = '/data/tf_workspace/model/dirty_mnist'
    LOG_TIME = datetime.datetime.now().strftime("%Y.%m.%d_%H:%M")

    transforms = A.Compose([
                    # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1),
                    A.HorizontalFlip(p=0.4),
                    A.VerticalFlip(p=0.4),
                    A.RandomRotate90(p=0.5),
                    # A.RandomBrightness(limit=0.1),
                    # A.RandomContrast(limit=0.2, p=0.5),
                ])

    TRAIN_DS_PATH = f'{DS_PATH}/dirty_mnist_2nd'
    df = pd.read_csv(f'{DS_PATH}/dirty_mnist_2nd_answer.csv')

    os.system('clear')
    images, labels, CLASSES = get_dataset()

    if not(os.path.isdir(f'/{SAVED_PATH}/{LOG_TIME}')):
        os.makedirs(f'/{SAVED_PATH}/{LOG_TIME}')
        f = open(f'{SAVED_PATH}/{LOG_TIME}/main_labels.txt', 'w')

        for label in CLASSES:
            f.write(f'{label}\n')
        
        f.close()

    train_cross_validate(images, labels)    