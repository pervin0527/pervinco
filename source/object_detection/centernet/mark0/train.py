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
    try:
        for res in pred:
            xmin, ymin, xmax, ymax, score, class_id = int(res[0]), int(res[1]), int(res[2]), int(res[3]), res[4], int(res[5])
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=(0, 0, 255))
            cv2.imwrite(f"./epoch_end/{class_id}_{score}.jpg")
    except:
        pass


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


def build_lrfn(lr_start=0.000001, lr_max=0.005, lr_min=0.00001, lr_rampup_epochs=1000, lr_sustain_epochs=0, lr_exp_decay=0.0001):
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
    train_data_dir = "/home/ubuntu/Datasets/WIDER/CUSTOM_TXT/augmentation/list.txt"
    test_data_dir = "/home/ubuntu/Datasets/WIDER/CUSTOM_TXT/test/list.txt"
    save_dir = "/home/ubuntu/Models"
    ckpt_path = "" #### /home/ubuntu/Models/centernet_resnet50_voc.h5

    backbone = "resnet50"
    classes = ["face"]
    epochs = 3000
    batch_size = 64
    max_detections = 10
    learning_rate = 1e-2
    input_shape = (512, 512, 3)
        
    train_dataset = Datasets(train_data_dir, input_shape=input_shape, batch_size=batch_size, num_classes=len(classes), max_detections=max_detections)
    test_dataset = Datasets(test_data_dir, input_shape=input_shape, batch_size=batch_size, num_classes=len(classes), max_detections=max_detections)

    train_steps = int(tf.math.ceil(len(train_dataset) / batch_size).numpy())
    test_steps = int(tf.math.ceil(len(test_dataset) / batch_size).numpy())

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    # optimizer = AngularGrad(method_angle="cos", learning_rate=learning_rate)
    cdr = tf.keras.optimizers.schedules.CosineDecayRestarts(initial_learning_rate=learning_rate,
                                                            first_decay_steps=300,
                                                            t_mul=2.0,
                                                            m_mul=0.9,
                                                            alpha=0.000001)
    callbacks = [DisplayCallback(),
                 tf.keras.callbacks.LearningRateScheduler(build_lrfn()),
                 tf.keras.callbacks.ModelCheckpoint(f"{save_dir}/centernet.h5", monitor="val_loss", verbose=1, save_best_only=True, save_weights_only=True)]

    with strategy.scope():
        model, prediction_model = centernet(input_shape=input_shape, num_classes=len(classes), backbone=backbone, max_objects=max_detections, mode="train") ## output shape : topk_x1, topk_y1, topk_x2, topk_y2, scores, class_ids
        
        if ckpt_path != '':
            model.load_weights(ckpt_path, by_name=True, skip_mismatch=True)

        for i in range(len(model.layers)):
            model.layers[i]. trainable = True

        model.compile(optimizer = optimizer, loss = {'centernet_loss': lambda y_true, y_pred: y_pred})
        # prediction_model.summary()
    
    model.fit(train_dataset,
              steps_per_epoch = train_steps,
              validation_data = test_dataset,
              validation_steps = test_steps,
              callbacks = callbacks,
              epochs = epochs)