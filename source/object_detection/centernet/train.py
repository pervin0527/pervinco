import os
import sys
import cv2
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from model import CenterNet
from dataloader import DataGenerator
from IPython.display import clear_output

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
np.set_printoptions(threshold=sys.maxsize)
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


class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        plot_predictions(model=model)


def plot_predictions(model):
    image = cv2.imread(f"{data_dir}/dog.jpg")
    image = cv2.resize(image, (input_shape[0], input_shape[1]))
    input_tensor = np.expand_dims(image, axis=0)
    hm_pred, wh_pred, reg_pred = model.predict(input_tensor)


if __name__ == "__main__":
    data_dir = "/home/ubuntu/Datasets/VOCdevkit/VOC2012/detection"
    label_dir = f"{data_dir}/Labels/labels.txt"
    
    backbone = "resnet18"
    epochs = 1000
    batch_size = 32
    threshold = 0.1
    max_detections = 100
    input_shape = [512, 512, 3]

    df = pd.read_csv(label_dir, sep=",", index_col=False, header=None)
    classes = df[0].to_list()
    
    train_dataset = DataGenerator(f"{data_dir}/train", "train", classes, batch_size, input_shape, max_detections)
    test_dataset = DataGenerator(f"{data_dir}/valid", "valid", classes, batch_size, input_shape, max_detections)
    train_steps = int(math.ceil(len(train_dataset) // batch_size))
    test_steps = int(math.ceil(len(test_dataset) // batch_size))
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    callbacks = [
        DisplayCallback(),
        tf.keras.callbacks.ModelCheckpoint("/home/ubuntu/Models/test.h5", monitor="val_loss", verbose=1, save_best_only=True, save_weights_only=True)
    ]

    model = CenterNet(input_shape, len(classes), max_detections, threshold, backbone, False)
    model.compile(optimizer=optimizer)
    model.fit(train_dataset,
              steps_per_epoch=train_steps,
              validation_data=test_dataset,
              validation_steps=test_steps,
              callbacks=callbacks,
              epochs=epochs)