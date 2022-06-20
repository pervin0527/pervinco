import os
import cv2
import numpy as np
import hparams as param
import tensorflow as tf

from PFLD import PFLD, PFLD_wing_loss_fn
from matplotlib import pyplot as plt
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

def get_overlay(image, landmarks):
    image = np.array(image).astype(np.uint8)
    for (x, y) in landmarks:
        cv2.circle(image, (int(x), int(y)), radius=1, color=(255, 0, 0), thickness=-1)

    return image


def plot_predictions(dataset, model):
    for item in dataset.take(1):
        image = item[0][6].numpy()

        image *= 128.0
        image += 128.0
        
        image_tensor = tf.expand_dims(image, axis=0)      
        prediction = model.predict(image_tensor, verbose=0)
        # landmarks = prediction[0].reshape(98, 2)
        landmarks = prediction.reshape(-1, 2)
        print(landmarks.shape)
        # landmarks = landmarks * [param.IMG_SIZE, param.IMG_SIZE]

        result_image = get_overlay(image, landmarks)
        plt.imshow(result_image)
        # plt.show()
        plt.imsave("train_epoch.png", result_image)


class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        plot_predictions(test_dataset, model=model)


def data_process(data):
    splits = tf.strings.split(data, sep=' ')
    image_path = splits[0]
    image_file = tf.io.read_file(image_path)
    image = tf.io.decode_jpeg(image_file, channels=3)
    
    image = tf.cast(image, dtype=tf.float32)
    image -= 128.0
    image /= 128.0
    
    image.set_shape([param.IMG_SIZE, param.IMG_SIZE, 3])

    landmarks = splits[1:197]
    landmarks = tf.strings.to_number(landmarks, out_type=tf.float32)
    landmarks.set_shape([param.N_LANDMARKS])
    
    attribute = splits[197:203]
    attribute = tf.strings.to_number(attribute, out_type=tf.float32)
    attribute.set_shape([6])
    
    euler_angle = splits[203:206]
    euler_angle = tf.strings.to_number(euler_angle, out_type=tf.float32)
    euler_angle.set_shape([3])

    return image, attribute, landmarks, euler_angle


def make_tf_data(txt_file):
    dataset = tf.data.TextLineDataset(txt_file)
    dataset = dataset.map(data_process, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size=param.BATCH_SIZE)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset

if __name__ == "__main__":
    train_dataset = make_tf_data(param.train_file_list)
    test_dataset = make_tf_data(param.test_file_list)

    print(train_dataset)
    print(test_dataset)

    with strategy.scope():
        optimizer = tf.keras.optimizers.Adam(learning_rate=param.LR)
        model = PFLD(input_size=param.IMG_SIZE, summary=False)
        # model = PFLD_wing_loss_fn(input_size=param.IMG_SIZE, summary=False)
        model.compile(optimizer=optimizer)

    TRAIN_STEPS_PER_EPOCH = int(tf.math.ceil(75000 / param.BATCH_SIZE).numpy())
    TEST_STEPS_PER_EPOCH = int(tf.math.ceil(2500 / param.BATCH_SIZE).numpy())

    callbacks = [DisplayCallback()]

    history = model.fit(
        train_dataset,
        steps_per_epoch=TRAIN_STEPS_PER_EPOCH,
        validation_data=test_dataset,
        validation_steps=TEST_STEPS_PER_EPOCH,
        verbose=1,
        callbacks=callbacks,
        epochs=param.EPOCHS
    )