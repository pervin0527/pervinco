import re
import math
import pathlib
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import KFold

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.mixed_precision import experimental as mixed_precision


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(gpus[0],
      [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=9000)])
  except RuntimeError as e:
    print(e)


MIXED_PRECISION = False
XLA_ACCELERATE = False

if MIXED_PRECISION:
    if tpu:
        policy = tf.keras.mixed_precision.experimental.Policy('mixed_bfloat16')
    else:
        policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
    mixed_precision.set_policy(policy)
    print('Mixed precision enabled')

if XLA_ACCELERATE:
    tf.config.optimizer.set_jit(True)
    print('Accelerated Linear Algebra enabled')


AUTO = tf.data.experimental.AUTOTUNE
strategy = tf.distribute.experimental.CentralStorageStrategy()
IMAGE_SIZE = [300, 300]
EPOCHS = 25
FOLDS = 3
BATCH_SIZE = 16 * strategy.num_replicas_in_sync
AUG_BATCH = BATCH_SIZE
FIRST_FOLD_ONLY = False
CLASSES = 4
SEED = 777

LR_START = 0.00001
LR_MAX = 0.00005 * strategy.num_replicas_in_sync
LR_MIN = 0.00001
LR_RAMPUP_EPOCHS = 5
LR_SUSTAIN_EPOCHS = 0
LR_EXP_DECAY = .8

def lrfn(epoch):
    if epoch < LR_RAMPUP_EPOCHS:
        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
        lr = LR_MAX
    else:
        lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY**(epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN
    return lr
    
lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose = True)


def basic_processing(ds_path, is_training):
    ds_path = pathlib.Path(ds_path)

    images = list(ds_path.glob('*/*'))
    images = [str(path) for path in images]

    if is_training:
        random.shuffle(images)

    labels = sorted(item.name for item in ds_path.glob('*/') if item.is_dir())
    # num_of_labels = len(labels)
    labels = dict((name, index) for index, name in enumerate(labels))
    labels = [labels[pathlib.Path(path).parent.name] for path in images]
    # labels = tf.keras.utils.to_categorical(labels, num_classes=num_of_labels, dtype='float32')

    return images, labels


def decode_image(image_data):
    image = tf.io.read_file(image_data)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0 
    image = tf.reshape(image, [*IMAGE_SIZE, 3])
    return image


def get_training_dataset(images, labels, do_aug=True):
    images = tf.data.Dataset.from_tensor_slices(images)
    images = images.map(decode_image, num_parallel_calls=AUTO)
    labels = tf.data.Dataset.from_tensor_slices(labels)

    dataset = tf.data.Dataset.zip((images, labels))
    dataset = dataset.repeat()
    dataset = dataset.batch(AUG_BATCH)

    if do_aug:
        dataset = dataset.map(transform, num_parallel_calls=AUTO)

    dataset = dataset.unbatch()
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO)

    return dataset


def onehot_encoding(label):
    return tf.one_hot(label, CLASSES)


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
    for j in range(AUG_BATCH):
        P = tf.cast( tf.random.uniform([],0,1)<=PROBABILITY, tf.int32)
        k = tf.cast( tf.random.uniform([],0,AUG_BATCH),tf.int32)
        x = tf.cast( tf.random.uniform([],0,DIM),tf.int32)
        y = tf.cast( tf.random.uniform([],0,DIM),tf.int32)
        b = tf.random.uniform([],0,1)
        WIDTH = tf.cast( DIM * tf.math.sqrt(1-b),tf.int32) * P
        ya = tf.math.maximum(0,y-WIDTH//2)
        yb = tf.math.minimum(DIM,y+WIDTH//2)
        xa = tf.math.maximum(0,x-WIDTH//2)
        xb = tf.math.minimum(DIM,x+WIDTH//2)
        
        one = image[j,ya:yb,0:xa,:]
        two = image[k,ya:yb,xa:xb,:]
        three = image[j,ya:yb,xb:DIM,:]
        middle = tf.concat([one,two,three],axis=1)
        img = tf.concat([image[j,0:ya,:,:],middle,image[j,yb:DIM,:,:]],axis=0)
        imgs.append(img)
        
        a = tf.cast(WIDTH*WIDTH/DIM/DIM,tf.float32)
        if len(label.shape)==1:
            lab1 = tf.one_hot(label[j],CLASSES)
            lab2 = tf.one_hot(label[k],CLASSES)
        else:
            lab1 = label[j,]
            lab2 = label[k,]
        labs.append((1-a)*lab1 + a*lab2)
            
    
    image2 = tf.reshape(tf.stack(imgs),(AUG_BATCH,DIM,DIM,3))
    label2 = tf.reshape(tf.stack(labs),(AUG_BATCH,CLASSES))
    return image2,label2


def mixup(image, label, PROBABILITY = 1.0):
    DIM = IMAGE_SIZE[0]
    
    imgs = []; labs = []
    for j in range(AUG_BATCH):
        P = tf.cast( tf.random.uniform([],0,1)<=PROBABILITY, tf.float32)
        k = tf.cast( tf.random.uniform([],0,AUG_BATCH),tf.int32)
        a = tf.random.uniform([],0,1)*P

        img1 = image[j,]
        img2 = image[k,]
        imgs.append((1-a)*img1 + a*img2)

        if len(label.shape)==1:
            lab1 = tf.one_hot(label[j], CLASSES)
            lab2 = tf.one_hot(label[k], CLASSES)
        else:
            lab1 = label[j,]
            lab2 = label[k,]
        labs.append((1-a)*lab1 + a*lab2)
            
    image2 = tf.reshape(tf.stack(imgs),(AUG_BATCH,DIM,DIM,3))
    label2 = tf.reshape(tf.stack(labs),(AUG_BATCH,CLASSES))
    return image2,label2


def transform(image,label):
    DIM = IMAGE_SIZE[0]
    SWITCH = 0.5
    CUTMIX_PROB = 0.666
    MIXUP_PROB = 0.666
    
    image2, label2 = cutmix(image, label, CUTMIX_PROB)
    image3, label3 = mixup(image, label, MIXUP_PROB)
    imgs = []; labs = []
    for j in range(AUG_BATCH):
        P = tf.cast( tf.random.uniform([],0,1)<=SWITCH, tf.float32)
        imgs.append(P*image2[j,]+(1-P)*image3[j,])
        labs.append(P*label2[j,]+(1-P)*label3[j,])
    
    image4 = tf.reshape(tf.stack(imgs),(AUG_BATCH,DIM,DIM,3))
    label4 = tf.reshape(tf.stack(labs),(AUG_BATCH,CLASSES))
    return image4,label4


def get_model():
    with strategy.scope():
        base_model = tf.keras.applications.EfficientNetB0(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
                                                          weights='imagenet',
                                                          include_top=False)
        base_model.trainable = True
        model = tf.keras.Sequential([base_model,
                                     tf.keras.layers.GlobalAveragePooling2D(),
                                     tf.keras.layers.Dense(CLASSES, activation='softmax', dtype='float32')])

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        return model


def train_cross_validate(images, labels, folds=5):
    histories = []
    models = []
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    kfold = KFold(folds, shuffle=True, random_state=SEED)
    
    for f, (train_index, valid_index) in enumerate(kfold.split(images, labels)):
        print('FOLD', f + 1)
        train_x, train_y = images[train_index[0] : (train_index[-1] + 1)], labels[train_index[0] : (train_index[-1] + 1)]
        valid_x, valid_y = images[valid_index[0] : (valid_index[-1] + 1)], labels[valid_index[0] : (valid_index[-1] + 1)]
        STEPS_PER_EPOCH = int(tf.math.ceil(len(images) / BATCH_SIZE).numpy())

        model = get_model()
        history = model.fit(get_training_dataset(train_x, train_y), 
                            steps_per_epoch = STEPS_PER_EPOCH,
                            epochs = EPOCHS,
                            callbacks = [lr_callback, early_stopping],
                            validation_data = get_validation_dataset(valid_x, valid_y),
                            verbose=1)
        models.append(model)
        histories.append(history)

        if FIRST_FOLD_ONLY:
            break

    return histories, models



if __name__ == "__main__":
    dataset = "/data/backup/pervinco_2020/Auged_datasets/test"
    images, labels = basic_processing(dataset, True)
    train_cross_validate(images, labels, folds=5)

    """ Display CutMix sample """
    # train_ds = get_training_dataset(train_images, train_labels, do_aug=False).unbatch()
    # augmented_element = train_ds.repeat().batch(AUG_BATCH).map(cutmix)

    # row = 6
    # col = 4
    # row = min(row, AUG_BATCH // col)

    # for (img,label) in augmented_element:
    #     plt.figure(figsize=(15,int(15*row/col)))
    #     for j in range(row*col):
    #         plt.subplot(row,col,j+1)
    #         plt.axis('off')
    #         plt.imshow(img[j,])
    #     plt.show()
    #     break

    """ Display MixUp sample """
    # row = 6; col = 4;
    # row = min(row,AUG_BATCH//col)
    # train_ds = get_training_dataset(train_images, train_labels, do_aug=False).unbatch()
    # augmented_element = train_ds.repeat().batch(AUG_BATCH).map(mixup)

    # for (img,label) in augmented_element:
    #     plt.figure(figsize=(15,int(15*row/col)))
    #     for j in range(row*col):
    #         plt.subplot(row,col,j+1)
    #         plt.axis('off')
    #         plt.imshow(img[j,])
    #     plt.show()
    #     break

    """ Display CutMix & MixUp sample """
    # row = 6; col = 4;
    # row = min(row,AUG_BATCH//col)
    # train_ds = get_training_dataset(train_images, train_labels, do_aug=False).unbatch()
    # augmented_element = train_ds.repeat().batch(AUG_BATCH).map(transform)

    # for (img,label) in augmented_element:
    #     plt.figure(figsize=(15,int(15*row/col)))
    #     for j in range(row*col):
    #         plt.subplot(row,col,j+1)
    #         plt.axis('off')
    #         plt.imshow(img[j,])
    #     plt.show()
    #     break