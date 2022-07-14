import os
import cv2
import numpy as np
import tensorflow as tf

from glob import glob
from model import centernet
from data_loader import Datasets
from optimizer import AngularGrad
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


def draw_result(idx, image, pred):
    scores = pred[:, 4]
    indices = np.where(scores > 0.7)
    print(scores)
    print(indices)


def plot_predictions(model):
    for idx, file in enumerate(sorted(glob("./samples/*"))):
        image = cv2.imread(file)
        image = cv2.resize(image, (input_shape[0], input_shape[1]))
        input_tensor = (image / 127.5) - 1
        input_tensor = np.expand_dims(input_tensor, axis=0)

        prediction = model.predict(input_tensor, verbose=0)[0]
        draw_result(idx, image, prediction)


class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)

        if not os.path.isdir("./epoch_end"):
            os.makedirs("./epoch_end")

        plot_predictions(model=prediction_model)


def build_lrfn(lr_start=0.000001, lr_max=0.001, lr_min=0.00001, lr_rampup_epochs=500, lr_sustain_epochs=0, lr_exp_decay=0.0001):
    # lr_max = lr_max * strategy.num_replicas_in_sync

    def lrfn(epoch):
        if epoch < lr_rampup_epochs:
            lr = (lr_max - lr_start) / lr_rampup_epochs * epoch + lr_start
        elif epoch < lr_rampup_epochs + lr_sustain_epochs:
            lr = lr_max
        else:
            lr = (lr_max - lr_min) * lr_exp_decay**(epoch - lr_rampup_epochs - lr_sustain_epochs) + lr_min
        
        return lr

    return lrfn


if __name__ == "__main__":
    train_data_dir = "/home/ubuntu/Datasets/WIDER/CUSTOM/augmentation/list.txt"
    test_data_dir = "/home/ubuntu/Datasets/WIDER/CUSTOM/test/list.txt"

    backbone = "resnet50"
    classes = ["face"]
    epochs = 1000
    batch_size = 64
    max_detections = 50
    learning_rate = 1e-4
    input_shape = (512, 512, 3)
        
    train_dataset = Datasets(train_data_dir, input_shape=input_shape, batch_size=batch_size, num_classes=len(classes), max_detections=max_detections)
    test_dataset = Datasets(test_data_dir, input_shape=input_shape, batch_size=batch_size, num_classes=len(classes), max_detections=max_detections)

    train_steps = int(tf.math.ceil(len(train_dataset) / batch_size).numpy())
    test_steps = int(tf.math.ceil(len(test_dataset) / batch_size).numpy())

    # optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    optimizer = AngularGrad(method_angle="cos", learning_rate=learning_rate)
    callbacks = [DisplayCallback(),
                 tf.keras.callbacks.LearningRateScheduler(build_lrfn()),]

    with strategy.scope():
        model, prediction_model = centernet(input_shape=input_shape, num_classes=len(classes), backbone=backbone, max_objects=max_detections, mode="train")
        ## output shape : topk_x1, topk_y1, topk_x2, topk_y2, scores, class_ids
        # prediction_model.summary()
        model.compile(optimizer = optimizer, loss = {'centernet_loss': lambda y_true, y_pred: y_pred})
    
    model.fit(train_dataset,
              steps_per_epoch = train_steps,
              validation_data = test_dataset,
              validation_steps = test_steps,
              callbacks = callbacks,
              epochs = epochs)