import os
import sys
import cv2
import math
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from model import centernet
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
        plot_predictions()


def preprocess_image(image):
    image = image.astype(np.float32)

    image[..., 0] -= 103.939
    image[..., 1] -= 116.779
    image[..., 2] -= 123.68

    return image


def plot_predictions():
    image = cv2.imread(f"./dog.jpg")
    image = cv2.resize(image, (input_shape[0], input_shape[1]))
    input_tensor = np.expand_dims(preprocess_image(image), axis=0)
    prediction = prediction_model.predict(input_tensor)[0]

    scores = prediction[:, 4]
    indices = np.where(scores > threshold)
    
    if len(indices) > 0:
        print(prediction[indices])


if __name__ == "__main__":
    data_dir = "/home/ubuntu/Datasets/VOCdevkit/VOC2012/detection"
    label_dir = f"{data_dir}/Labels/labels.txt"
    
    backbone = "resnet50"
    freeze_backbone = True

    epochs = 1000
    init_lr = 0.0001
    max_lr = 0.009
    batch_size = 64
    threshold = 0.4
    max_detections = 100
    input_shape = [512, 512, 3]

    df = pd.read_csv(label_dir, sep=",", index_col=False, header=None)
    classes = df[0].to_list()
    
    train_dataset = DataGenerator(f"{data_dir}/train", "train", classes, batch_size, input_shape, max_detections)
    test_dataset = DataGenerator(f"{data_dir}/valid", "valid", classes, batch_size, input_shape, max_detections)
    train_steps = int(math.ceil(len(train_dataset) // batch_size))
    test_steps = int(math.ceil(len(test_dataset) // batch_size))
    
    optimizer = tf.keras.optimizers.Adam()
    clr = tfa.optimizers.CyclicalLearningRate(initial_learning_rate=init_lr,
                                              maximal_learning_rate=max_lr,
                                              scale_fn=lambda x : 1.0,
                                              step_size=epochs / 2)
    callbacks = [
        DisplayCallback(),
        tf.keras.callbacks.LearningRateScheduler(clr),
        tf.keras.callbacks.ModelCheckpoint("/home/ubuntu/Models/test.h5", monitor="val_loss", verbose=1, save_best_only=True, save_weights_only=True)
    ]

    with strategy.scope():
        model, prediction_model = centernet(len(classes), backbone, input_shape[0], max_detections, threshold, False, False, True)

        if freeze_backbone:
            for i in range(190):
                model.layers[i].trainable = False
        else:
            model.load_weights("/home/ubuntu/Models/test.h5", by_name=True, skip_mismatch=True)
        model.summary()
        model.compile(optimizer=optimizer, loss={'centernet_loss': lambda y_true, y_pred: y_pred})

    model.fit(train_dataset,
              steps_per_epoch=train_steps,
              validation_data=test_dataset,
              validation_steps=test_steps,
              callbacks=callbacks,
              epochs=epochs)