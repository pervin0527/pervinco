import random, re, math, os, gc, pathlib, datetime
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf, tensorflow.keras.backend as K
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
        tf.config.experimental.set_virtual_device_configuration(gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=11000)])
  except RuntimeError as e:
    # 프로그램 시작시에 메모리 증가가 설정되어야만 합니다
    print(e)

strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
AUTOTUNE = tf.data.experimental.AUTOTUNE
IMG_HEIGHT = 270
IMG_WIDTH = 480
NUM_EPOCHS = 20
BATCH_SIZE = 3 * strategy.num_replicas_in_sync
SEED = 100
LR = 0.0001
NUM_OF_CLASSES = 1049


def basic_processing(ds_path, labels_list, labels_len, is_training):
    ds_path = pathlib.Path(ds_path)

    images = list(ds_path.glob('*/*'))
    images = [str(path) for path in images]

    if is_training:
        random.shuffle(images)

    labels = dict((name, index) for index, name in enumerate(labels_list))
    labels = [labels[pathlib.Path(path).parent.name] for path in images]
    labels = tf.keras.utils.to_categorical(labels, num_classes=labels_len, dtype='float32')

    return images, labels


def preprocess_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])

    NEW_IMAGE_SIZE = [int(image.shape[0]), int(image.shape[1])]

    image = tf.cast(image, tf.float32) / 255.0
    image = tf.reshape(image, [*NEW_IMAGE_SIZE, 3])
    
    return image


def make_dataset(images, labels):
    image_ds = tf.data.Dataset.from_tensor_slices(images)
    image_ds = image_ds.map(preprocess_image, num_parallel_calls=AUTOTUNE)
    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(labels, tf.float32))
    image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))

    return image_label_ds


def get_lr_callback():
    lr_start   = 0.000001*10*0.5
    lr_max     = 0.0000005 * BATCH_SIZE * 10 * 0.5
    lr_min     = 0.000001 * 10*0.5
    lr_ramp_ep = 5
    lr_sus_ep  = 0
    lr_decay   = 0.8
   
     
    def lrfn(epoch):
        if epoch < lr_ramp_ep:
            lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start   
        elif epoch < lr_ramp_ep + lr_sus_ep:
            lr = lr_max    
        else:
            lr = (lr_max - lr_min) * lr_decay**(epoch - lr_ramp_ep - lr_sus_ep) + lr_min    
        return lr

    lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose = False)
    return lr_callback


def get_model():
    with strategy.scope():
        input_layer = tf.keras.layers.Input(shape = (None, None, 3))
        base_model = tf.keras.applications.EfficientNetB6(weights="imagenet", include_top=False)(input_layer)
        gap = tf.keras.layers.GlobalAveragePooling2D()(base_model)

        dense = tf.keras.layers.Dense(NUM_OF_CLASSES)(gap)
        prelu = tf.keras.layers.PReLU()(dense)
        output = tf.keras.layers.Softmax(dtype="float32", name="softmax")(prelu)

        model = tf.keras.Model(inputs=input_layer, outputs=output)

        optimizer = tf.keras.optimizers.Adam(learning_rate = LR)
        model.compile(optimizer=optimizer, loss=[tf.keras.losses.CategoricalCrossentropy()], metrics=[tf.keras.metrics.CategoricalAccuracy()])

    return model


def load_all_models():
    all_models = []
    model_names = ['1_landmark_cls.h5', '2_landmark_cls.h5', '3_landmark_cls.h5', '4_landmark_cls.h5', '5_landmark_cls.h5']
    for model_name in model_names:
        file_name = os.path.join('/data/tf_workspace/models/landmark_classification/k-fold', model_name)
        model = tf.keras.models.load_model(file_name)
        all_models.append(model)
        print('Model loaded : ', file_name)

    return all_models


def ensemble_model(models):
    inputs = tf.keras.Input(shape = (None, None, 3))
    for i, model in enumerate(models):
        for layer in model.layers:
            layer.trainable = False
    ensemble_outputs = [model(inputs) for model in models]
    merge = tf.keras.layers.concatenate(ensemble_outputs)
    output = tf.keras.layers.Dense(NUM_OF_CLASSES, activation='softmax')(merge)
    ensemble_model = tf.keras.models.Model(inputs=inputs, outputs=output)   
    tf.keras.utils.plot_model(ensemble_model, "/data/tf_workspace/models/landmark_classification/ensemble.png")

    return ensemble_model


def train_cross_validate(images, labels, folds = 5):
    kf = KFold(n_splits=5, random_state=0, shuffle=True)

    for fold, (train_index, valid_index) in enumerate(kf.split(images, labels)):
        print('FOLD', fold + 1)
        
        train_x, train_y = images[train_index[0] : (train_index[-1] + 1)], labels[train_index[0] : (train_index[-1] + 1)]
        valid_x, valid_y = images[valid_index[0] : (valid_index[-1] + 1)], labels[valid_index[0] : (valid_index[-1] + 1)]

        TRAIN_STEP_PER_EPOCH = int(tf.math.ceil(len(train_x) / BATCH_SIZE).numpy())
        VALID_STEP_PER_EPOCH = int(tf.math.ceil(len(valid_x) / BATCH_SIZE).numpy())

        train_ds = make_dataset(train_x, train_y)
        valid_ds = make_dataset(valid_x, valid_y)
        train_ds = train_ds.repeat().shuffle(2048).batch(BATCH_SIZE).prefetch(AUTOTUNE)
        valid_ds = valid_ds.repeat().shuffle(2048).batch(BATCH_SIZE).prefetch(AUTOTUNE)

        model = get_model()

        # cb_lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', 
        #                                                     mode = 'min', 
        #                                                     factor = 0.9, 
        #                                                     patience = 1, 
        #                                                     verbose = 1, 
        #                                                     min_delta = 0.0001)


        saved_path = '/data/tf_workspace/models/landmark_classification'
        time = datetime.datetime.now().strftime("%Y.%m.%d_%H:%M") + '_tf2'

        if not(os.path.isdir(saved_path + '/' + time)):
            os.makedirs(os.path.join(saved_path + '/' + time))

        checkpoint_path = saved_path + '/' + time + '/' + str(fold + 1) + '-{epoch:02d}-{val_categorical_accuracy:.2f}.hdf5'
        cb_checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                            monitor='val_categorical_accuracy',
                                                            save_best_only=True,
                                                            mode='max')

        history = model.fit(train_ds,
                            epochs=NUM_EPOCHS,
                            steps_per_epoch=TRAIN_STEP_PER_EPOCH,
                            validation_data=valid_ds,
                            validation_steps=VALID_STEP_PER_EPOCH,
                            verbose=1,
                            callbacks=[cb_checkpointer, get_lr_callback()])

        model.save(saved_path + '/' + time + '/' + str(fold + 1) + '_' + 'landmark_cls.h5')

    train_x, valid_x, train_y, valid_y = train_test_split(images, labels, test_size=0.2)
    print(len(train_x), len(train_y))

    train_ds = make_dataset(train_x, train_y)
    valid_ds = make_dataset(valid_x, valid_y)
    train_ds = train_ds.repeat().shuffle(2048).batch(BATCH_SIZE).prefetch(AUTOTUNE)
    valid_ds = valid_ds.repeat().shuffle(2048).batch(BATCH_SIZE).prefetch(AUTOTUNE)

    TRAIN_STEP_PER_EPOCH = int(tf.math.ceil(len(train_x) / BATCH_SIZE).numpy())
    VALID_STEP_PER_EPOCH = int(tf.math.ceil(len(valid_x) / BATCH_SIZE).numpy())

    with strategy.scope():
        models = load_all_models()
        model = ensemble_model(models)

        optimizer = tf.keras.optimizers.Adam(learning_rate = LR)
        model.compile(optimizer=optimizer, loss=[tf.keras.losses.CategoricalCrossentropy()], metrics=[tf.keras.metrics.CategoricalAccuracy()])

    history = model.fit(train_ds,
                        epochs=NUM_EPOCHS,
                        steps_per_epoch=TRAIN_STEP_PER_EPOCH,
                        validation_data=valid_ds,
                        validation_steps=VALID_STEP_PER_EPOCH,
                        verbose=1,
                        callbacks=[get_lr_callback()])

    model.save('/data/tf_workspace/models/landmark_classification/k-fold/ensemble_landmark_cls.h5')


if __name__ == "__main__":
    # DATA_SET_PATH = '/data/tf_workspace/datasets/public'
    # TRAIN_DS = tf.io.gfile.glob(DATA_SET_PATH + '/train/*')
    # TEST_DS = tf.io.gfile.glob(DATA_SET_PATH + '/test/*')

    DS_PATH = '/data/tf_workspace/datasets/public/train'
    labels_csv = '/data/tf_workspace/datasets/public/category.csv'
    labels_df = pd.read_csv(labels_csv)
    labels_list = labels_df['landmark_name'].tolist()

    images, labels = basic_processing(DS_PATH, labels_list, len(labels_list), True)
    train_cross_validate(images, labels)