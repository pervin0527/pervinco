import tensorflow as tf
import pandas as pd
import pathlib
import random
import os
import datetime
import time

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        print("True")
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

AUTOTUNE = tf.data.experimental.AUTOTUNE
strategy = tf.distribute.experimental.CentralStorageStrategy()

BATCH_SIZE = 8
IMG_HEIGHT = 270
IMG_WIDTH = 480
NUM_EPOCHS = 1000
EARLY_STOP_PATIENCE = 3
LR = 0.0001

def basic_processing(ds_path, labels_list, labels_len, is_training):
    ds_path = pathlib.Path(ds_path)

    images = list(ds_path.glob('*/*'))
    images = [str(path) for path in images]
    len_images = len(images)

    if is_training:
        random.shuffle(images)

    labels = dict((name, index) for index, name in enumerate(labels_list))
    labels = [labels[pathlib.Path(path).parent.name] for path in images]
    labels = tf.keras.utils.to_categorical(labels, num_classes=labels_len, dtype='float32')

    return images, labels, len_images, labels_len


def preprocess_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    
    # image = tf.keras.applications.efficientnet.preprocess_input(image)
    
    return image


def make_tf_dataset(images, labels):
    image_ds = tf.data.Dataset.from_tensor_slices(images)
    image_ds = image_ds.map(preprocess_image, num_parallel_calls=AUTOTUNE)
    lable_ds = tf.data.Dataset.from_tensor_slices(tf.cast(labels, tf.float32))
    image_label_ds = tf.data.Dataset.zip((image_ds, lable_ds))

    return image_label_ds


def build_lrfn(lr_start=0.000001*10*0.5, lr_max=0.0000005 * BATCH_SIZE * 10*0.5, 
               lr_min=0.000001 * 10*0.5, lr_rampup_epochs=5, 
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

model_name = 'Efficientnet-B0'
dataset_name = 'landmark_classification'
train_dataset_path = '/data/backup/pervinco_2020/Auged_datasets/' + dataset_name + '/train'
valid_dataset_path = '/data/backup/pervinco_2020/Auged_datasets/' + dataset_name + '/valid'

labels_csv = '/data/backup/pervinco_2020/datasets/data/public/category.csv'
labels_df = pd.read_csv(labels_csv)
labels_list = labels_df['landmark_name'].tolist()

train_images, train_labels, train_images_len, train_labels_len = basic_processing(train_dataset_path, labels_list, len(labels_list), True)
valid_images, valid_labels, valid_images_len, valid_labels_len = basic_processing(valid_dataset_path, labels_list, len(labels_list), False)

TRAIN_STEP_PER_EPOCH = int(tf.math.ceil(train_images_len / BATCH_SIZE).numpy())
VALID_STEP_PER_EPOCH = int(tf.math.ceil(valid_images_len / BATCH_SIZE).numpy())

saved_path = '/data/backup/pervinco_2020/model/'
time = datetime.datetime.now().strftime("%Y.%m.%d_%H:%M") + '_tf2'
weight_file_name = '{epoch:02d}-{val_categorical_accuracy:.2f}.hdf5'

if not(os.path.isdir(saved_path + dataset_name + '/' + time)):
    os.makedirs(os.path.join(saved_path + dataset_name + '/' + time))

    f = open(saved_path + dataset_name + '/' + time + '/README.txt', 'w')
    f.write(train_dataset_path + '\n')
    f.write(valid_dataset_path + '\n')
    f.write(str(IMG_HEIGHT) + '\n')
    f.write(str(IMG_WIDTH) + '\n')
    f.write("Model : " + model_name)
    f.close()

else:
    pass

train_ds = make_tf_dataset(train_images, train_labels)
valid_ds = make_tf_dataset(valid_images, valid_labels)

train_ds = train_ds.repeat().batch(BATCH_SIZE).prefetch(AUTOTUNE)
valid_ds = valid_ds.repeat().batch(BATCH_SIZE).prefetch(AUTOTUNE)

# input_layer = tf.keras.layers.Input(shape = (IMG_HEIGHT, IMG_WIDTH, 3))
base_model = tf.keras.applications.EfficientNetB3(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
                                                  weights="imagenet", # noisy-student
                                                  include_top=False)
gap = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)

dense = tf.keras.layers.Dense(train_labels_len)(gap)
prelu = tf.keras.layers.PReLU()(dense)
output = tf.keras.layers.Softmax(dtype="float32", name="softmax")(prelu)

model = tf.keras.Model(inputs=base_model.input, outputs=output)

optimizer = tf.keras.optimizers.Adam(learning_rate = LR)
model.compile(optimizer=optimizer, loss=[tf.keras.losses.CategoricalCrossentropy()], metrics=[tf.keras.metrics.CategoricalAccuracy()])
model.summary()

cb_early_stopper = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
checkpoint_path = saved_path + dataset_name + '/' + time + '/' + weight_file_name
cb_checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                        monitor='val_categorical_accuracy',
                                                        save_best_only=True,
                                                        mode='max')
lrfn = build_lrfn()
lr_schedule = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=1)    

history = model.fit(train_ds,
                    epochs=NUM_EPOCHS,
                    steps_per_epoch=TRAIN_STEP_PER_EPOCH,
                    validation_data=valid_ds,
                    validation_steps=VALID_STEP_PER_EPOCH,
                    verbose=1,
                    callbacks=[cb_early_stopper, cb_checkpointer, lr_schedule])

model.save(saved_path + dataset_name + '/' + time + '/' + dataset_name + '.h5')