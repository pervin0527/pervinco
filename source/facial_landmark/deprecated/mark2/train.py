import os
import cv2
import numpy as np
import tensorflow as tf
from glob import glob
from model import MobileNet
from IPython.display import clear_output
from tensorflow.python.keras import backend as K

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
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


def get_overlay(index, image, landmarks):
    image = np.array(image).astype(np.uint8)
    for (x, y) in landmarks:
        cv2.circle(image, (int(x), int(y)), radius=1, color=(0, 0, 255), thickness=-1)

    cv2.imwrite(f"epochs/epoch_{index}.png", image)


def plot_predictions(model):
    for idx, file in enumerate(sorted(glob("./samples/sample*"))):
        image = tf.io.read_file(file)
        image_tensor = tf.image.decode_jpeg(image, channels=1)
        image_tensor = tf.image.resize(image_tensor, (IMG_SIZE, IMG_SIZE))
        image_tensor = image_tensor / 255.0
        image_tensor = tf.expand_dims(image_tensor, axis=0)

        prediction = model.predict(image_tensor, verbose=0)

        rgb_image = cv2.imread(file)
        resized_image = cv2.resize(rgb_image, (IMG_SIZE, IMG_SIZE))        
        landmark = prediction[0] * IMG_SIZE
        landmark = landmark.reshape(-1, 2)
        
        get_overlay(idx, resized_image, landmark)


class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)

        if not os.path.isdir("./epochs"):
            os.makedirs("./epochs")
        plot_predictions(model=model)


def smoothL1(y_true, y_pred):
    x = K.abs(y_true - y_pred)
    x = K.switch(x < HUBER_DELTA, 0.5 * x ** 2, HUBER_DELTA * (x - 0.5 * HUBER_DELTA))
    return  K.sum(x)


def data_process(data):
    splits = tf.strings.split(data, sep=' ')
    image_path = splits[0]
    image_file = tf.io.read_file(image_path)
    image = tf.io.decode_jpeg(image_file, channels=1)
    
    image = tf.cast(image, dtype=tf.float32)
    image = image / 255.0
    image.set_shape(INPUT_SHAPE)

    # label = splits[1:206]
    label = splits[1:137]
    label = tf.strings.to_number(label, out_type=tf.float32)

    return image, label


def build_dataset(txt_file):
    n_dataset = '/'.join(txt_file.split('/')[:-1])
    n_dataset = len(glob(f"{n_dataset}/images/*"))

    dataset = tf.data.TextLineDataset(txt_file)
    dataset = dataset.map(data_process, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.repeat()
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset, n_dataset


def build_model():
    # base_model = tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape=INPUT_SHAPE, include_top=False, weights=None, pooling="max")
    base_model = MobileNet(input_shape=(64, 64, 1), alpha=1.0, depth_multiplier=1, include_top=False, weights=None, pooling='max', shallow=True)
    last_layer = base_model.get_layer("global_max_pooling2d").output
    out = tf.keras.layers.Dense(N_LANDMARKS)(last_layer)

    model = tf.keras.Model(base_model.input, out)

    if print_summary:
        model.summary()

    return model


if __name__ == "__main__":
    EPOCHS = 3000
    BATCH_SIZE = 2048
    IMG_SIZE = 64
    INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, 1)
    HUBER_DELTA = 0.5
    N_LANDMARKS = 68 * 2
    LEARNING_RATE = 1e-3

    print_summary = False
    
    train_data = "/data/Source/PFLD_68points_Pytorch/data/300VW/train_data/list.txt"
    test_data = "/data/Source/PFLD_68points_Pytorch/data/WFLW/test_data/list.txt"

    train_dataset, n_train_dataset = build_dataset(train_data)
    test_dataset, n_test_dataset = build_dataset(test_data)
    print(n_train_dataset, n_test_dataset)
    print(train_dataset)
    print(test_dataset)

    train_steps = int(n_train_dataset / BATCH_SIZE)
    test_steps = int(n_test_dataset / BATCH_SIZE)

    callbacks = [DisplayCallback(),
                 tf.keras.callbacks.ModelCheckpoint("/data/Models/test/best.ckpt", monitor="val_loss", verbose=1, save_best_only=True, save_weights_only=True)]

    model = build_model()
    model.compile(loss=smoothL1, optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), metrics=["mape"])
    model.fit(train_dataset, steps_per_epoch=train_steps, epochs=EPOCHS, verbose=1, validation_data=test_dataset, validation_steps=test_steps, callbacks=callbacks)