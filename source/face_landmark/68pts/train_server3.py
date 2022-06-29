import os
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from glob import glob
from losses import PFLDLoss
from data import PFLDDatasets
from model import PFLDInference
from IPython.display import clear_output

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="2"
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
        cv2.circle(image, (int(x), int(y)), radius=1, color=(255, 255, 0), thickness=-1)

    cv2.imwrite(f"epochs/server3_{index}.png", image)


def plot_predictions(model):
    for idx, file in enumerate(sorted(glob("./samples/*"))):
        image = tf.io.read_file(file)
        image_tensor = tf.image.decode_jpeg(image, channels=3)
        image_tensor = tf.image.resize(image_tensor, (input_shape[0], input_shape[1]))
        image_tensor = image_tensor / 255.0
        image_tensor = tf.expand_dims(image_tensor, axis=0)

        prediction = model.predict(image_tensor, verbose=0)
        pred1, pred2 = prediction[0], prediction[1]
        # print(pred1.shape, pred2.shape) # 199, 196

        rgb_image = cv2.imread(file)
        height, width = rgb_image.shape[:2]
        
        landmark = pred2 * input_shape[0]
        landmark[0::2] = landmark[0::2] * width / input_shape[0]
        landmark[1::2] = landmark[1::2] * height / input_shape[0]
        landmark = landmark.reshape(-1, 2)
        
        get_overlay(idx, rgb_image, landmark)


class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)

        if not os.path.isdir("./epochs"):
            os.makedirs("./epochs")
        plot_predictions(model=model)


def data_process(data):
    splits = tf.strings.split(data, sep=' ')
    image_path = splits[0]
    image_file = tf.io.read_file(image_path)
    image = tf.io.decode_jpeg(image_file, channels=3)
    
    image = tf.cast(image, dtype=tf.float32)
    image = image / 255.0
    image.set_shape((112, 112, 3))

    label = splits[1:146]
    label = tf.strings.to_number(label, out_type=tf.float32)

    return image, label


def build_dataset(txt_file):
    n_dataset = '/'.join(txt_file.split('/')[:-1])
    n_dataset = len(glob(f"{n_dataset}/imgs/*"))

    dataset = tf.data.TextLineDataset(txt_file)
    dataset = dataset.map(data_process, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset, n_dataset


if __name__ == "__main__":
    train_dir = '/home/ubuntu/Datasets/WFLW/train_data_68pts/list.txt'
    test_dir = '/home/ubuntu/Datasets/WFLW/test_data_68pts/list.txt'
    save_dir = "/home/ubuntu/Models/face_landmark_68pts3"

    batch_size = 128
    epochs = 10000
    model_path = ''
    input_shape = [112, 112, 3]
    lr = 0.001

    train_datasets, n_train_datasets = build_dataset(train_dir)
    valid_datasets, n_valid_datasets = build_dataset(test_dir)

    train_steps_per_epoch = int(n_train_datasets / batch_size)
    valid_steps_per_epoch = int(n_valid_datasets / batch_size)

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    optimizer = tf.keras.optimizers.Adam()
    cdr = tf.keras.optimizers.schedules.CosineDecayRestarts(initial_learning_rate=lr,
                                                            first_decay_steps=200,
                                                            t_mul=1.0,
                                                            m_mul=0.9,
                                                            alpha=0.00001)
    
    callback = [DisplayCallback(),
                tf.keras.callbacks.LearningRateScheduler(cdr),
                tf.keras.callbacks.ModelCheckpoint(f"{save_dir}/best.h5", monitor="val_loss", verbose=1, save_best_only=True, save_weights_only=True)]

    with strategy.scope():
        model = PFLDInference(input_shape, is_train=True, keypoints=68*2)

        if model_path != '':
            model.load_weights(model_path, by_name=True, skip_mismatch=True)
            print("WEIGHT LOADED")

        model.compile(loss={'train_out': PFLDLoss()}, optimizer=optimizer)
    
    history = model.fit(x=train_datasets,
                        validation_data=valid_datasets,
                        epochs=epochs,
                        callbacks=callback,
                        steps_per_epoch=train_steps_per_epoch,
                        validation_steps=valid_steps_per_epoch,
                        verbose=1)