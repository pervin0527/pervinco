import os, re, datetime, math, pathlib, random, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.backend as K
import matplotlib
matplotlib.use('Agg')

from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import KFold
from tensorflow.keras.mixed_precision import experimental as mixed_precision


gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 1:
    try:
        print("Activate Multi GPU")
        for gpu in gpus:
            tf.config.experimental.set_virtual_device_configuration(gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10000)])
    except RuntimeError as e:
        print(e)

else:
    try:
        print("Activate Sigle GPU")
        tf.config.experimental.set_memory_growth(gpus[0], True)
        strategy = tf.distribute.experimental.CentralStorageStrategy()
    except RuntimeError as e:
        print(e)


AUTO = tf.data.experimental.AUTOTUNE
strategy = tf.distribute.experimental.CentralStorageStrategy()
BATCH_SIZE = 4 * strategy.num_replicas_in_sync
EPOCHS = 1000
# IMAGE_SIZE = [380, 380]
IMAGE_SIZE = [224, 224]


def basic_processing(ds_path, is_training):
    ds_path = pathlib.Path(ds_path)

    images = list(ds_path.glob('*/*'))
    images = [str(path) for path in images]

    if is_training:
        random.shuffle(images)

    labels = sorted(item.name for item in ds_path.glob('*/') if item.is_dir())
    num_of_labels = len(labels)
    labels = dict((name, index) for index, name in enumerate(labels))
    labels = [labels[pathlib.Path(path).parent.name] for path in images]
    # labels = tf.keras.utils.to_categorical(labels, num_classes=num_of_labels, dtype='float32')

    return images, labels, num_of_labels


def decode_image(image_data):
    image = tf.io.read_file(image_data)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    # image = tf.cast(image, tf.float32) / 255.0
    image = tf.keras.applications.efficientnet.preprocess_input(image)
    image = tf.reshape(image, [*IMAGE_SIZE, 3])
    return image


def onehot_encoding(label):
    return tf.one_hot(label, CLASSES)


def get_training_dataset(images, labels, do_aug=True):
    images = tf.data.Dataset.from_tensor_slices(images)
    images = images.map(decode_image, num_parallel_calls=AUTO)
    labels = tf.data.Dataset.from_tensor_slices(labels)

    dataset = tf.data.Dataset.zip((images, labels))
    dataset = dataset.repeat()
    dataset = dataset.batch(BATCH_SIZE)

    if do_aug:
        dataset = dataset.map(transform, num_parallel_calls=AUTO)

    dataset = dataset.unbatch()
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO)

    return dataset


def get_validation_dataset(images, labels, do_onehot=True):
    images = tf.data.Dataset.from_tensor_slices(images)
    images = images.map(decode_image, num_parallel_calls=AUTO)
    labels = tf.data.Dataset.from_tensor_slices(labels)
    labels = labels.map(onehot_encoding, num_parallel_calls=AUTO)
    
    dataset = tf.data.Dataset.zip((images, labels))
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.cache()
    dataset = dataset.prefetch(AUTO)

    return dataset


def cutmix(image, label, PROBABILITY = 1.0):
    DIM = IMAGE_SIZE[0]
    
    imgs = []; labs = []
    for j in range(BATCH_SIZE):
        P = tf.cast( tf.random.uniform([], 0,1)<=PROBABILITY, tf.int32)
        k = tf.cast( tf.random.uniform([], 0, BATCH_SIZE),tf.int32)
        x = tf.cast( tf.random.uniform([], 0, DIM),tf.int32)
        y = tf.cast( tf.random.uniform([], 0, DIM),tf.int32)
        b = tf.random.uniform([],0,1)
        WIDTH = tf.cast( DIM * tf.math.sqrt(1 - b), tf.int32) * P
        ya = tf.math.maximum(0, y - WIDTH // 2)
        yb = tf.math.minimum(DIM, y + WIDTH // 2)
        xa = tf.math.maximum(0, x - WIDTH // 2)
        xb = tf.math.minimum(DIM, x + WIDTH // 2)
        
        one = image[j, ya : yb, 0 : xa, :]
        two = image[k, ya : yb, xa : xb, :]
        three = image[j, ya : yb, xb : DIM, :]
        middle = tf.concat([one, two, three], axis=1)
        img = tf.concat([image[j, 0 : ya, : , :], middle, image[j, yb : DIM, :, :]], axis=0)
        imgs.append(img)
        
        a = tf.cast(WIDTH * WIDTH / DIM / DIM, tf.float32)
        if len(label.shape)==1:
            lab1 = tf.one_hot(label[j], CLASSES)
            lab2 = tf.one_hot(label[k], CLASSES)
        else:
            lab1 = label[j,]
            lab2 = label[k,]
        labs.append((1 - a) * lab1 + a * lab2)
            
    
    image2 = tf.reshape(tf.stack(imgs),(BATCH_SIZE, DIM,DIM, 3))
    label2 = tf.reshape(tf.stack(labs),(BATCH_SIZE, CLASSES))
    return image2,label2


def mixup(image, label, PROBABILITY = 1.0):
    DIM = IMAGE_SIZE[0]
    
    imgs = []; labs = []
    for j in range(BATCH_SIZE):
        P = tf.cast( tf.random.uniform([], 0,1) <= PROBABILITY, tf.float32)
        k = tf.cast( tf.random.uniform([], 0, BATCH_SIZE),tf.int32)
        a = tf.random.uniform([],0,1)*P

        img1 = image[j,]
        img2 = image[k,]
        imgs.append((1 - a) * img1 + a * img2)

        if len(label.shape)==1:
            lab1 = tf.one_hot(label[j], CLASSES)
            lab2 = tf.one_hot(label[k], CLASSES)
        else:
            lab1 = label[j,]
            lab2 = label[k,]
        labs.append((1 - a) * lab1 + a * lab2)
            
    image2 = tf.reshape(tf.stack(imgs),(BATCH_SIZE, DIM,DIM, 3))
    label2 = tf.reshape(tf.stack(labs),(BATCH_SIZE, CLASSES))
    return image2,label2


def transform(image,label):
    DIM = IMAGE_SIZE[0]
    SWITCH = 0.5
    CUTMIX_PROB = 0.666
    MIXUP_PROB = 0.666
    
    image2, label2 = cutmix(image, label, CUTMIX_PROB)
    image3, label3 = mixup(image, label, MIXUP_PROB)
    imgs = []; labs = []
    for j in range(BATCH_SIZE):
        P = tf.cast( tf.random.uniform([], 0, 1) <= SWITCH, tf.float32)
        imgs.append(P*image2[j,] + (1 - P) * image3[j,])
        labs.append(P*label2[j,] + (1 - P) * label3[j,])
    
    image4 = tf.reshape(tf.stack(imgs),(BATCH_SIZE, DIM,DIM, 3))
    label4 = tf.reshape(tf.stack(labs),(BATCH_SIZE, CLASSES))
    return image4,label4


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
        base_model = tf.keras.applications.EfficientNetB4(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
                                                          weights='imagenet',
                                                          include_top=False)
        base_model.trainable = True
        model = tf.keras.Sequential([base_model,
                                     tf.keras.layers.GlobalAveragePooling2D(),
                                     tf.keras.layers.Dense(CLASSES, activation='softmax', dtype='float32')])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
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

        WEIGHT_FNAME = '{epoch:02d}-{val_categorical_accuracy:.2f}.hdf5'
        checkpoint_path = f'{SAVED_PATH}/{LOG_TIME}/{f+1}-{WEIGHT_FNAME}'
        cb_checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                             monitor='val_categorical_accuracy',
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

        model.save(f'{SAVED_PATH}/{LOG_TIME}/{f+1}_cut_mix_up.h5')

        models.append(model)
        histories.append(history)

    return histories, models


def visualize(augmentation_element, name):
    row, col = 6, 4
    row = min(row, BATCH_SIZE // col)

    for (image, label) in augmentation_element:
        image = image / 255.0
        plt.figure(figsize=(15, int(15 * row / col)))
        for j in range(row * col):
            plt.subplot(row, col, j + 1)
            plt.axis('off')
            plt.imshow(image[j, ])

        plt.savefig(f'{SAVED_PATH}/{LOG_TIME}/{name}.jpg')
        plt.show()
        break


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cut Mix & Mix UP")
    parser.add_argument('--input_dataset', type=str)
    parser.add_argument('--visualize', type=str2bool, default=False)
    args = parser.parse_args()

    DATASET_PATH = args.input_dataset
    DATASET_NAME = DATASET_PATH.split('/')[-1]
    SAVED_PATH = f'/data/tf_workspace/model/{DATASET_NAME}'
    LOG_TIME = datetime.datetime.now().strftime("%Y.%m.%d_%H:%M")

    images, labels, CLASSES = basic_processing(DATASET_PATH, True)

    if not(os.path.isdir(f'/{SAVED_PATH}/{LOG_TIME}')):
        os.makedirs(f'/{SAVED_PATH}/{LOG_TIME}')


    if args.visualize == True:
        train_ds = get_training_dataset(images, labels, do_aug=False).unbatch()
        cut_mix_element = train_ds.repeat().batch(BATCH_SIZE).map(cutmix)
        visualize(cut_mix_element, 'cut_mix')
        del cut_mix_element

        mix_up_element = train_ds.repeat().batch(BATCH_SIZE).map(mixup)
        visualize(mix_up_element, 'mix_up')
        del mix_up_element

        merged_element = train_ds.repeat().batch(BATCH_SIZE).map(transform)
        visualize(merged_element, 'merged')
        del merged_element


    histories, models = train_cross_validate(images, labels, folds=5)
    
    for h, m in zip(histories, models):
        print(h, m)
