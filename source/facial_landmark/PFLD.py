import os
import cv2
import math
import numpy as np
import pandas as pd
import tensorflow as tf

from glob import glob
from IPython.display import clear_output
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


def get_files(dir):
    image_files = sorted(glob(f"{dir}/images/*.jpg"))
    keypoint_files = sorted(glob(f"{dir}/keypoints/*.txt"))

    print(len(image_files), len(keypoint_files))
    return image_files, keypoint_files


def data_process(images, keypoints):
    images = tf.io.read_file(images)
    images = tf.image.decode_jpeg(images, channels=3)
    images = tf.cast(images, dtype=tf.float32)
    images -= 128.0
    images /= 128.0
    images.set_shape([IMG_SIZE, IMG_SIZE, 3])

    keypoints = keypoints / IMG_SIZE
    keypoints.set_shape([LANDMARKS])
  
    return images, keypoints


def load_keypoints(keypoint_files):
    total_keypoints = []

    for file in keypoint_files:
        f = open(file, "r")
        keypoints = f.readline()
        keypoints = keypoints.split(',')
        keypoints = list(map(float, keypoints))

        total_keypoints.append(keypoints)

    return np.array(total_keypoints)


def get_tf_data(images, keypoints, is_train):
    dataset = tf.data.Dataset.from_tensor_slices((images, keypoints))
    dataset = dataset.map(data_process, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.repeat()
    if is_train:
        dataset = dataset.shuffle(buffer_size=len(train_image_files))
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def read_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, size=[IMG_SIZE, IMG_SIZE])

    return image


def show_predictions(image_list, model):
    if not os.path.isdir("./Display_callback"):
        os.makedirs("./Display_callback")
        
    for idx, image_file in enumerate(image_list):
        result_image = cv2.imread(image_file)
        image_tensor = read_image(image_file)
        predictions = model.predict(np.expand_dims(image_tensor, axis=0), verbose=0)
        predictions = predictions.reshape((-1, 98, 2)) * IMG_SIZE

        for (x, y) in predictions[0]:
            cv2.circle(result_image, (int(x), int(y)), radius=1, color=(0, 0, 255), thickness=3)

        cv2.imwrite(f"./Display_callback/{idx}.jpg", result_image)


class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        show_predictions(test_image_files[:5], model=model)


def _inverted_res_block(inputs, expansion, stride, filters, use_res_connect, stage=1, block_id=1, expand=True, output2=False):
    in_channels = tf.keras.backend.int_shape(inputs)[-1]
    x = inputs
    name = 'bbn_stage{}_block{}'.format(stage, block_id)

    if expand:
        x = tf.keras.layers.Conv2D(expansion*in_channels, kernel_size=1, padding='same', use_bias=False, name=name + '_expand_conv')(x)
        x = tf.keras.layers.BatchNormalization(epsilon=1e-3, momentum=0.999, name=name + '_expand_bn')(x)
        x = tf.keras.layers.ReLU(name=name + 'expand_relu')(x)

    out2 = x

    # Depthwise
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=stride, use_bias=False, padding='same', name=name+'_dw_conv')(x)
    x = tf.keras.layers.BatchNormalization(epsilon=1e-3, momentum=0.999, name=name+'_dw_bn')(x)
    x = tf.keras.layers.ReLU(name=name + '_dw_relu')(x)

    # Project
    x = tf.keras.layers.Conv2D(filters, kernel_size=1, padding='same', activation=None, use_bias=False, name=name + '_project_conv')(x)
    x = tf.keras.layers.BatchNormalization(epsilon=1e-3, momentum=0.999, name=name + '_project_bn')(x)

    if use_res_connect:
        return tf.keras.layers.Add(name=name+'_add')([inputs, x])
    
    if output2:
        return x, out2
    
    return x


def PFLD():
    input_layer = tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    x = tf.keras.layers.Conv2D(64, kernel_size=3, strides=2, padding='same', use_bias=False, name='conv1')(input_layer)
    x = tf.keras.layers.BatchNormalization(name='conv1_bn')(x)
    x = tf.keras.layers.ReLU(name='conv1_relu')(x)

    x = tf.keras.layers.Conv2D(64, kernel_size=3, strides=1, padding='same', use_bias=False, name='conv2')(x)
    x = tf.keras.layers.BatchNormalization(name='conv2_bn')(x)
    x = tf.keras.layers.ReLU(name='conv2_relu')(x)
    
    x = _inverted_res_block(x, expansion=2, stride=2, filters=64, use_res_connect=False, stage=3, block_id=1)
    
    x = _inverted_res_block(x, expansion=2, stride=1, filters=64, use_res_connect=True, stage=3, block_id=2)
    x = _inverted_res_block(x, expansion=2, stride=1, filters=64, use_res_connect=True, stage=3, block_id=3)
    x = _inverted_res_block(x, expansion=2, stride=1, filters=64, use_res_connect=True, stage=3, block_id=4)
    out1 = _inverted_res_block(x, expansion=2, stride=1, filters=64, use_res_connect=True, stage=3, block_id=5)

    x = _inverted_res_block(out1, expansion=2, stride=2, filters=128, use_res_connect=False, stage=4, block_id=1)
    
    x = _inverted_res_block(x, expansion=2, stride=1, filters=128, use_res_connect=False, stage=5, block_id=1)

    x = _inverted_res_block(x, expansion=4, stride=1, filters=128, use_res_connect=True, stage=5, block_id=2)
    x = _inverted_res_block(x, expansion=4, stride=1, filters=128, use_res_connect=True, stage=5, block_id=3)
    x = _inverted_res_block(x, expansion=4, stride=1, filters=128, use_res_connect=True, stage=5, block_id=4)
    x = _inverted_res_block(x, expansion=4, stride=1, filters=128, use_res_connect=True, stage=5, block_id=5)
    x = _inverted_res_block(x, expansion=4, stride=1, filters=128, use_res_connect=True, stage=5, block_id=6)

    # 16 x 14 x 14
    x = _inverted_res_block(x, expansion=2, stride=1, filters=16, use_res_connect=False, stage=6, block_id=1)

    x1 = tf.keras.layers.AvgPool2D(14)(x)
    x1 = tf.keras.layers.Flatten()(x1)

    # 32 x 7 x 7
    x = tf.keras.layers.Conv2D(32, kernel_size=3, strides=2, padding='same', use_bias=False, name='conv7')(x)
    x = tf.keras.layers.BatchNormalization(name='conv7_bn')(x)
    x = tf.keras.layers.ReLU(name='conv7_relu')(x)

    x2 = tf.keras.layers.AvgPool2D(7)(x)
    x2 = tf.keras.layers.Flatten()(x2)

    x = tf.keras.layers.Conv2D(128, kernel_size=7, strides=1, padding='valid', use_bias=False, name='conv_8')(x)
    x = tf.keras.layers.BatchNormalization(name='conv8_bn')(x)
    x = tf.keras.layers.ReLU(name='conv8_relu')(x)

    x3 = tf.keras.layers.Flatten()(x)
    
    multi_scale = tf.keras.layers.concatenate([x1, x2, x3], axis=1)
    landmarks = tf.keras.layers.Dense(LANDMARKS)(multi_scale)
    
    # model = tf.keras.models.Model(inputs=input_layer, outputs=[out1, landmarks])
    model = tf.keras.models.Model(inputs=input_layer, outputs=landmarks)
    return model


def wing_loss_fn(w=10.0, epsilon=2.0):
    def wing_loss(y_true, y_pred):
        y_pred = tf.reshape(y_pred, (-1, 98, 2))
        y_true = tf.reshape(y_true, (-1, 98, 2))

        x = y_true - y_pred
        c = w * (1.0 - tf.math.log(1.0 + w / epsilon))
        absolute_x = tf.abs(x)
        losses = tf.where(w > absolute_x, w * tf.math.log(1.0 + absolute_x / epsilon), absolute_x - c)
        loss = tf.reduce_mean(tf.reduce_sum(losses, axis=[1, 2]), axis=0)
        return loss
    return wing_loss


if __name__ == "__main__":
    EPOCHS = 300
    IMG_SIZE = 224
    BATCH_SIZE = 32
    LANDMARKS = 98 * 2
    SAVE_DIR = "/data/Models/facial_landmark"

    train_dir = "/data/Datasets/WFLW/train"
    test_dir = "/data/Datasets/WFLW/test"

    train_image_files, train_keypoint_files = get_files(train_dir)
    test_image_files, test_keypoint_files = get_files(test_dir)

    train_keypoints = load_keypoints(train_keypoint_files)
    test_keypoints = load_keypoints(test_keypoint_files)
    print(train_keypoints.shape, test_keypoints.shape)

    train_dataset = get_tf_data(train_image_files, train_keypoints, is_train=True)
    test_dataset = get_tf_data(test_image_files, test_keypoints, is_train=False)
    print(train_dataset)
    print(test_dataset)

    train_steps_per_epoch = int(tf.math.ceil(len(train_image_files) / BATCH_SIZE).numpy())
    test_steps_per_epoch = int(tf.math.ceil(len(test_image_files) / BATCH_SIZE).numpy())

    callbacks = [DisplayCallback(),
                 tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, verbose=1),
                 tf.keras.callbacks.ModelCheckpoint(f"{SAVE_DIR}/best.ckpt", monitor="val_loss", verbose=1, mode="min", save_best_only=True, save_weights_only=True)]

    model = PFLD()
    model.summary()
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001), loss = wing_loss_fn()) # tf.keras.losses.MeanSquaredError()

    history = model.fit(
        train_dataset,
        steps_per_epoch=train_steps_per_epoch,
        validation_data=test_dataset,
        validation_steps=test_steps_per_epoch,
        verbose=1,
        epochs=EPOCHS,
        callbacks=callbacks
    )