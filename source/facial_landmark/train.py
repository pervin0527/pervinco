import os
import cv2
import numpy as np
import tensorflow as tf

from glob import glob
from losses import PFLDLoss
from data import PFLDDatasets
from model import PFLDInference
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


def get_overlay(index, image, landmarks):
    image = np.array(image).astype(np.uint8)
    for (x, y) in landmarks:
        cv2.circle(image, (int(x), int(y)), radius=1, color=(0, 0, 255), thickness=-1)

    image = cv2.resize(image, (320, 320))
    cv2.imwrite(f"epochs/epoch_{index}.png", image)


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


def adjust_lr(epoch, lr):
    if epoch < 3:
        return lr
    else:
        return lr * 0.93


def build_lrfn(lr_start=0.001, lr_max=0.007, 
               lr_min=0.0001, lr_rampup_epochs=5, 
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


if __name__ == "__main__":   
    batch_size = 256
    epochs = 1000
    model_path = ''
    input_shape = [112, 112, 3]
    lr = 0.001
    
    train_datasets = PFLDDatasets('/data/Datasets/CUSTOM_WFLW/train_data/list.txt', batch_size)
    valid_datasets = PFLDDatasets('/data/Datasets/CUSTOM_WFLW/test_data/list.txt', batch_size)
    
    lrfn = build_lrfn()
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    callback = [DisplayCallback(),
                tf.keras.callbacks.LearningRateScheduler(lrfn),
                tf.keras.callbacks.ModelCheckpoint("/data/Models/facial_landmark/best.h5", monitor="val_loss", verbose=1, save_best_only=True, save_weights_only=True)]

    with strategy.scope():
        model = PFLDInference(input_shape, is_train=True)

        if model_path != '':
            model.load_weights(model_path, by_name=True, skip_mismatch=True)

        model.compile(loss={'train_out': PFLDLoss()}, optimizer=optimizer)
    
    history = model.fit(x=train_datasets,
                        validation_data=valid_datasets,
                        workers=1,
                        epochs=epochs,
                        callbacks=callback,
                        steps_per_epoch=len(train_datasets),
                        validation_steps=len(valid_datasets),
                        verbose=1)