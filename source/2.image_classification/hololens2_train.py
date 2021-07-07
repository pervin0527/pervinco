import os, cv2, datetime, pathlib, random, argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_io as tfio
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K

# K.set_image_data_format("channels_first")

# GPU setup
gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 1:
    try:
        print("Activate Multi GPU")
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


def preprocess_image(images, label=None):
    image = tf.io.read_file(images)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tfio.experimental.color.rgb_to_bgr(image)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    # image = tf.transpose(image, perm=(2, 0, 1))

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
    classes = labels
    labels = dict((name, index) for index, name in enumerate(labels))
    labels = [labels[pathlib.Path(path).parent.name] for path in images]
    labels = tf.keras.utils.to_categorical(labels, num_classes=len(classes), dtype='float32')

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

    return dataset, total_images, classes


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


# def get_model():
#     inputs = tf.keras.Input(shape=(3, IMG_SIZE, IMG_SIZE), name="input_2")

#     base_model = tf.keras.applications.EfficientNetB0(include_top=False,
#                                                       weights="imagenet",
#                                                       pooling="avg")(inputs)

#     base_model.trainable = True

#     x = tf.keras.layers.Dropout(0.2, name='top_dropout')(base_model)
#     outputs = tf.keras.layers.Dense(len(train_classes), activation="softmax", name="Identity")(x)
#     model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
#     model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics = ['categorical_accuracy'])
#     model.summary()

#     return model


## BEST performance
def get_model():
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="input_2")
    base_model = tf.keras.applications.EfficientNetB1(weights="imagenet", # noisy-student
                                                      include_top=False)(inputs)

    base_model.trainable = True

    x = tf.keras.layers.GlobalAveragePooling2D()(base_model)
    outputs = tf.keras.layers.Dense(len(train_classes), activation="softmax", name="Identity")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics = ['categorical_accuracy'])
    model.summary()

    return model


# def get_model():
#     inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="input_2")
#     x = tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5, offset=-1)(inputs)

#     base_model = tf.keras.applications.EfficientNetB0(include_top=False,
#                                                       weights="imagenet",
#                                                       pooling="avg")(x)

#     base_model.trainable = True

#     x = tf.keras.layers.Dropout(0.2, name='top_dropout')(base_model)
#     outputs = tf.keras.layers.Dense(len(train_classes), activation="softmax", name="Identity")(x)
#     model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
#     model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics = ['categorical_accuracy'])
#     model.summary()

#     return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image classification model Training")
    parser.add_argument('--input_dataset', type=str)
    parser.add_argument('--visualize', type=str2bool, default=False)
    args = parser.parse_args()

    dataset = args.input_dataset
    DATASET_NAME = dataset.split('/')[-2]
    TRAIN_PATH = f'{dataset}/train'
    VALID_PATH = f'{dataset}/valid'
    # print(TRAIN_PATH, VALID_PATH)

    # Load data & Set hyper-parameters
    AUTO = tf.data.experimental.AUTOTUNE
    EPOCHS = 10
    BATCH_SIZE = 64 * strategy.num_replicas_in_sync
    IMG_SIZE = 224

    train_dataset, train_total, train_classes = make_tf_dataset(TRAIN_PATH, True)
    valid_dataset, valid_total, valid_classes = make_tf_dataset(VALID_PATH, False)

    # for item in train_dataset.take(1):
    #     print(item)

    TRAIN_STEPS_PER_EPOCH = int(tf.math.ceil(train_total/ BATCH_SIZE).numpy())
    VALID_STEP_PER_EPOCH = int(tf.math.ceil(valid_total / BATCH_SIZE).numpy())

    print(len(train_classes), len(valid_classes))

    if args.visualize == True:
        for i in range(3):
            tf_data_visualize(train_dataset)
            tf_data_visualize(valid_dataset)
    
    # Learning Rate Scheduler setup
    lrfn = build_lrfn()
    lr_schedule = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=1)

    # Checkpoint callback setup
    SAVED_PATH = f'/data/Models/{DATASET_NAME}'
    LOG_TIME = datetime.datetime.now().strftime("%Y.%m.%d_%H:%M")
    WEIGHT_FNAME = '{epoch:02d}-{val_categorical_accuracy:.2f}.hdf5'
    checkpoint_path = f'/{SAVED_PATH}/{LOG_TIME}/{WEIGHT_FNAME}'

    if not(os.path.isdir(f'/{SAVED_PATH}/{LOG_TIME}')):
        os.makedirs(f'/{SAVED_PATH}/{LOG_TIME}')
        f = open(f'{SAVED_PATH}/{LOG_TIME}/main_labels.txt', 'w')

        for label in train_classes:
            f.write(f'{label}\n')
        pooling="avg"
        f.close()

    checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                      monitor='val_loss',
                                                      save_best_only=True,
                                                      mode='min')
    earlystopper = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

    model = get_model()    
    history = model.fit(train_dataset,
                        epochs=EPOCHS,
                        callbacks=[lr_schedule, checkpointer, earlystopper],
                        steps_per_epoch=TRAIN_STEPS_PER_EPOCH,
                        verbose=1,
                        validation_data=valid_dataset,
                        validation_steps=VALID_STEP_PER_EPOCH)

    # tf.saved_model.save(model, f'{SAVED_PATH}/{LOG_TIME}/saved_model')
    tf.keras.models.save_model(model, f'{SAVED_PATH}/{LOG_TIME}/saved_model')

    display_training_curves(history)

    os.system(f'python3 -m tf2onnx.convert --saved-model {SAVED_PATH}/{LOG_TIME}/saved_model --opset 9 --output {SAVED_PATH}/{LOG_TIME}/converted.onnx --inputs-as-nchw input_2:0')
    # os.system(f'python3 -m tf2onnx.convert --saved-model {SAVED_PATH}/{LOG_TIME}/saved_model --opset 9 --output {SAVED_PATH}/{LOG_TIME}/converted.onnx')