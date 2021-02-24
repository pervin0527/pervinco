import cv2, pathlib, datetime, os, random
import numpy as np
import pandas as pd
import tensorflow as tf
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform

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


def get_dataset(df):
    CLASSES = [c for c in df]
    CLASSES = CLASSES[1:]

    # print(len(df))
    X = np.zeros([len(df), IMG_SIZE, IMG_SIZE, 3], dtype=np.uint8)
    y = np.zeros([len(df), len(CLASSES)], dtype=np.uint8)

    for idx in tqdm(range(len(df))):
        file_name = str(df.iloc[idx, 0]).zfill(5)
        image = cv2.imread(f'{TRAIN_DS_PATH}/{file_name}.png')

        image2 = np.where((image <= 254) & (image != 0), 0, image)
        X[idx] = image2

        label = df.iloc[idx, 1:].values.astype('float')
        y[idx] = label

    return X, y, CLASSES


def normalize_image(image, label):
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

    return dataset


def get_model():
    with strategy.scope():
        base_model = tf.keras.applications.EfficientNetB6(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                                                          weights='imagenet', # noisy-student
                                                          include_top=False)
        base_model.trainable = True
            
        avg = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
        output = tf.keras.layers.Dense(26, activation="sigmoid")(avg)
        model = tf.keras.Model(inputs=base_model.input, outputs=output)

    model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['binary_accuracy'])
    
    return model


def split_dataset():
    df = pd.read_csv(f'{DS_PATH}/dirty_mnist_2nd_answer.csv')
    kfold = KFold(n_splits=N_FOLD)

    for fold, (train, valid) in enumerate(kfold.split(df, df.index)):
        df.loc[valid, 'kfold'] = int(fold)

    if not(os.path.isdir(f'{DS_PATH}/custom_split')):
        os.makedirs(f'{DS_PATH}/custom_split')

    df.to_csv(f'{DS_PATH}/custom_split/split_kfold.csv', index=False)


def train_cross_validate():

    split_dataset()
    df = pd.read_csv(f'{DS_PATH}/custom_split/split_kfold.csv')

    if not(os.path.isdir(f'/{SAVED_PATH}/{LOG_TIME}')):
        os.makedirs(f'/{SAVED_PATH}/{LOG_TIME}')

    os.system('clear')

    for i in range(N_FOLD):
        df_train = df[df['kfold'] != i].reset_index(drop=True)
        df_valid = df[df['kfold'] == i].reset_index(drop=True)

        df_train.drop(['kfold'], axis=1).to_csv(f'{DS_PATH}/custom_split/train-kfold-{i}.csv', index=False)
        df_valid.drop(['kfold'], axis=1).to_csv(f'{DS_PATH}/custom_split/valid-kfold-{i}.csv', index=False)

        df_train = pd.read_csv(f'{DS_PATH}/custom_split/train-kfold-{i}.csv')
        df_valid = pd.read_csv(f'{DS_PATH}/custom_split/valid-kfold-{i}.csv')

        train_x, train_y, train_classes = get_dataset(df_train)
        valid_x, valid_y, valid_classes = get_dataset(df_valid)

        print('FOLD', i + 1)

        output_path = f'/{SAVED_PATH}/{LOG_TIME}/{i+1}'
        os.makedirs(output_path)
        print(train_x.shape, train_y.shape, valid_x.shape, valid_y.shape)

        WEIGHT_FNAME = '{epoch:02d}-{val_binary_accuracy:.2f}.hdf5'
        checkpoint_path = f'{output_path}/{i+1}-{WEIGHT_FNAME}'
        cb_checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                             monitor='val_binary_accuracy',
                                                             save_best_only=True,
                                                             mode='max')
        cb_early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

        TRAIN_STEPS_PER_EPOCH = int(tf.math.ceil(len(train_x) / BATCH_SIZE).numpy())
        VALID_STEPS_PER_EPOCH = int(tf.math.ceil(len(valid_x) / BATCH_SIZE).numpy())

        model = get_model()
        model.fit(make_tf_dataset(train_x, train_y), 
                  steps_per_epoch = TRAIN_STEPS_PER_EPOCH,
                  epochs = EPOCHS,
                  validation_data = make_tf_dataset(valid_x, valid_y),
                  validation_steps = VALID_STEPS_PER_EPOCH,
                  verbose=1,
                  callbacks = [cb_checkpointer, cb_early_stopping])

        model.save(f'{output_path}/{i+1}_dmnist.h5')

        del train_x, train_y
        del valid_x, valid_y


def self_cut_mix(img):
    transformed_img = img.copy()
    l = random.randrange(img.shape[0] // 8, img.shape[0] // 2 + 1)

    # cut
    if random.choice([True, False]):
        i1 = random.randrange(l, img.shape[0] - l + 1)
        j1 = random.randrange(0, img.shape[0] - l + 1)
        i2 = random.randrange(0, i1 - l + 1)
        j2 = random.randrange(0, img.shape[0] - l + 1)
        
    else:
        i1 = random.randrange(0, img.shape[0] - l + 1)
        j1 = random.randrange(l, img.shape[0] - l + 1)
        i2 = random.randrange(0, img.shape[0] - l + 1)
        j2 = random.randrange(0, j1 - l + 1)

    p1 = img[i1:i1+l, j1:j1+l].copy()
    p2 = img[i2:i2+l, j2:j2+l].copy()

    # rotate piece
    random_rotate_90 = A.RandomRotate90(p=1)
    p1 = random_rotate_90(image=p1)["image"]
    p2 = random_rotate_90(image=p2)["image"]

    # mix
    if random.choice([True, False]):
        transformed_img[i1:i1+l, j1:j1+l] = p2
        transformed_img[i2:i2+l, j2:j2+l] = p1

    else:
        transformed_img[i1:i1+l, j1:j1+l] = p1
        transformed_img[i2:i2+l, j2:j2+l] = p2

    return transformed_img


class SelfCutMix(ImageOnlyTransform):
    def __init__(self, always_apply=False, num_pieces=[3, 4, 5], p=1):
        super(SelfCutMix, self).__init__(always_apply, p)

    def apply(self, img, **params):
        return self_cut_mix(img)


def shuffle_pieces(img, num_pieces: list):
    n = random.choice(num_pieces)
    transformed_img = img.copy()
    l = img.shape[0] // n
    pieces = [transformed_img[h:h+l, w:w+l].copy() for h in range(0, n*l, l) for w in range(0, n*l, l)]

    random.shuffle(pieces)
    random_rotate_90 = A.RandomRotate90(p=1)
    for i, h in enumerate(range(0, n*l, l)):
        for j, w in enumerate(range(0, n*l, l)):
            transformed_img[h:h+l, w:w+l] = random_rotate_90(image=pieces[i*n+j])["image"]

    return transformed_img


class ShufflePieces(ImageOnlyTransform):
    def __init__(self, always_apply=False, num_pieces=[3, 4, 5], p=1):
        super(ShufflePieces, self).__init__(always_apply, p)
        self.num_pieces = num_pieces

    def apply(self, img, **params):
        return shuffle_pieces(img, self.num_pieces)


if __name__ == "__main__":
    EPOCHS = 100
    IMG_SIZE = 256
    IMAGE_SIZE = [IMG_SIZE, IMG_SIZE]
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    BATCH_SIZE = 10 * strategy.num_replicas_in_sync
    N_FOLD = 5
    DS_PATH = '/data/tf_workspace/datasets/dirty_mnist_2'
    SAVED_PATH = '/data/tf_workspace/model/dirty_mnist'
    TRAIN_DS_PATH = f'{DS_PATH}/dirty_mnist_2nd'
    LOG_TIME = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")

    transform = A.Compose([
        A.RandomRotate90(p=1),

        A.OneOf([A.HorizontalFlip(p=0.7),
                 A.VerticalFlip(p=0.7),], p=0.7),

        A.OneOf([ShufflePieces(p=0.7),
                 SelfCutMix(0.7),], p=0.7)
    ])

    train_cross_validate()    
