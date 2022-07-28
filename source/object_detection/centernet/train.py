import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

from glob import glob
from model import centernet
from optimizer import AngularGrad
from data_loader import DataGenerator
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


def preprocess_image(image):
    image = image.astype(np.float32)

    image[..., 0] -= 103.939
    image[..., 1] -= 116.779
    image[..., 2] -= 123.68

    return image


def draw_result(idx, image, detections):
    scores = detections[:, 4]
    indices = np.where(scores > 0.7)[0]
    # print(indices, detections[indices])
    detections[:, [0, 2]] = np.clip(detections[:, [0, 2]], 0, image.shape[1])
    detections[:, [1, 3]] = np.clip(detections[:, [1, 3]], 0, image.shape[0])

    if len(indices):
        result_image = image.copy()
        result_image = cv2.resize(result_image, (input_shape[0] // 4, input_shape[1] // 4))
        for result in detections[indices]:
            xmin, ymin, xmax, ymax, score, label = int(result[0]), int(result[1]), int(result[2]), int(result[3]), result[4], int(result[5])
            cv2.rectangle(result_image, (xmin, ymin), (xmax, ymax), (0, 0, 255))

        cv2.imwrite("./epoch_end/result.jpg", result_image)


def plot_predictions(model):
    for idx, file in enumerate(sorted(glob("./samples/*"))):
        image = cv2.imread(file)
        image = cv2.resize(image, (input_shape[0], input_shape[1]))
        input_tensor = preprocess_image(image)
        # input_tensor = (image / 127.5) - 1
        input_tensor = np.expand_dims(input_tensor, axis=0)

        prediction = model.predict(input_tensor, verbose=0)[0]
        draw_result(idx, image, prediction)


class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)

        if not os.path.isdir("./epoch_end"):
            os.makedirs("./epoch_end")

        plot_predictions(model=prediction_model)


def build_lrfn(lr_start=0.00001, lr_max=0.001, lr_min=0.000001, lr_rampup_epochs=150, lr_sustain_epochs=0, lr_exp_decay=0.0001):
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
    root_dir = "/home/ubuntu/Datasets/WIDER"
    train_data_dir = f"{root_dir}/FACE2/train_512"
    test_data_dir = f"{root_dir}/FACE2/test_512"

    epochs = 300
    batch_size = 64
    max_detections = 10
    input_shape = (512, 512, 3)
    backbone = "resnet50"
    freeze_backbone = True
    save_dir = "/home/ubuntu/Models/CenterNet"
    label_file = f"{root_dir}/Labels/labels.txt"
    label_file = pd.read_csv(label_file, sep=',', index_col=False, header=None)
    classes = label_file[0].tolist()
    print(classes)

    if freeze_backbone:
        learning_rate = 0.001
        save_name = f"{save_dir}/freeze.h5"
        ckpt_path = ""

    else:
        learning_rate = 0.0001
        save_name = f"{save_dir}/unfreeze.h5"
        ckpt_path = f"{save_dir}/freeze.h5"
    
    train_generator = DataGenerator(train_data_dir,
                                    'list',
                                    classes=classes,
                                    skip_difficult=True,
                                    skip_truncated=True,
                                    multi_scale=False,
                                    batch_size=batch_size,
                                    shuffle_groups=False,
                                    input_size=input_shape[0])

    test_generator = DataGenerator(test_data_dir,
                                   'list',
                                   classes=classes,
                                   skip_difficult=True,
                                   skip_truncated=True,
                                   multi_scale=False,
                                   batch_size=batch_size,
                                   shuffle_groups=False,
                                   input_size=input_shape[0])


    optimizer = AngularGrad(method_angle="cos", learning_rate=learning_rate)
    alpha = learning_rate * 0.1
    cdr = tf.keras.optimizers.schedules.CosineDecayRestarts(initial_learning_rate=learning_rate,
                                                            first_decay_steps=epochs,
                                                            t_mul=1.0,
                                                            m_mul=1.0,
                                                            alpha=alpha)

    callbacks = [DisplayCallback(),
                 tf.keras.callbacks.LearningRateScheduler(build_lrfn()),
                 tf.keras.callbacks.TensorBoard(log_dir=f"{save_dir}", update_freq='epoch'),
                 tf.keras.callbacks.ModelCheckpoint(save_name, monitor="val_loss", verbose=1, save_best_only=True, save_weights_only=True)]

    with strategy.scope():
        model, prediction_model = centernet(input_shape=input_shape, num_classes=len(classes), backbone=backbone, max_detections=max_detections, mode="train", freeze_bn=freeze_backbone)

        if freeze_backbone:
            for i in range(190):
                model.layers[i].trainable = False
                # print(model.layers[i].name)
        else:
            model.load_weights(ckpt_path, by_name=True, skip_mismatch=True)

            for layer in model.layers:
                layer.trainable = True

        model.compile(optimizer = optimizer, loss = {'centernet_loss': lambda y_true, y_pred: y_pred})
        # prediction_model.summary()
    
    model.fit(train_generator,
              steps_per_epoch = int(tf.math.ceil(train_generator.size() / batch_size).numpy()),
              validation_data = test_generator,
              validation_steps = int(tf.math.ceil(test_generator.size() / batch_size).numpy()),
              callbacks = callbacks,
              epochs = epochs)