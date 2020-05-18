import tensorflow as tf
from tensorflow import keras
import pathlib
import random
import time

AUTOTUNE = tf.data.experimental.AUTOTUNE
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)


def basic_processing(ds_path, is_training):
    ds_path = pathlib.Path(ds_path)

    images = list(ds_path.glob('*/*'))
    images = [str(path) for path in images]
    len_images = len(images)

    if is_training:
        random.shuffle(images)

    labels = sorted(item.name for item in ds_path.glob('*/') if item.is_dir())
    labels_len = len(labels)
    labels = dict((name, index) for index, name in enumerate(labels))
    labels = [labels[pathlib.Path(path).parent.name] for path in images]

    return images, labels, len_images, labels_len


def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = keras.applications.xception.preprocess_input(image)

    return image

def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)

def make_tf_dataset(images, labels):
    image_ds = tf.data.Dataset.from_tensor_slices(images)
    image_ds = image_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)

    lable_ds = tf.data.Dataset.from_tensor_slices(tf.cast(labels, tf.int64))

    image_label_ds = tf.data.Dataset.zip((image_ds, lable_ds))

    return image_label_ds


if __name__ == "__main__":
    train_dataset_path = '/data/backup/pervinco_2020/datasets/cu50/train5'
    valid_dataset_path = '/data/backup/pervinco_2020/datasets/cu50/valid5'

    train_images, train_labels, train_images_len, train_labels_len = basic_processing(train_dataset_path, True)
    valid_images, valid_labels, valid_images_len, valid_labels_len = basic_processing(valid_dataset_path, False)

    BATCH_SIZE = 32
    IMG_SIZE = 224
    TRAIN_STEP_PER_EPOCH = tf.math.ceil(train_images_len / BATCH_SIZE).numpy()
    VALID_STEP_PER_EPOCH = tf.math.ceil(valid_images_len / BATCH_SIZE).numpy()

    train_ds = make_tf_dataset(train_images, train_labels)
    valid_ds = make_tf_dataset(valid_images, valid_labels)

    train_ds = train_ds.repeat().batch(BATCH_SIZE)
    train_ds = train_ds.prefetch(1)
    valid_ds = valid_ds.repeat().batch(BATCH_SIZE)
    valid_ds = valid_ds.prefetch(1)

    base_model = keras.applications.xception.Xception(input_shape=(224, 224, 3),
                                                  weights="imagenet",
                                                  include_top=False)
    avg = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    output = tf.keras.layers.Dense(train_labels_len, activation="softmax")(avg)
    model = tf.keras.Model(inputs=base_model.input, outputs=output)

    for layer in base_model.layers:
        layer.trainable = True

    optimizer = tf.keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    cb_early_stopper = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

    history = model.fit(train_ds,
                        epochs=10,
                        steps_per_epoch=TRAIN_STEP_PER_EPOCH,
                        shuffle=False,
                        validation_data=valid_ds,
                        validation_steps=VALID_STEP_PER_EPOCH,
                        verbose=1,
                        callbacks=[cb_early_stopper]
                        )

    model.save('/data/backup/pervinco_2020/model/test_model/tf_data_test_model')
