import os
import tensorflow as tf
import tensorflow_addons as tfa

from functools import partial
from model import yolo_base, get_train_model
from loss import get_yolo_loss
from data_loader import DataGenerator
from data_utils import read_label_file, read_txt_file

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


if __name__ == "__main__":
    phi = "s"
    epoch = 100
    batch_size = 16
    max_detections = 500
    weight_decay = 5e-4
    input_shape = [640, 640]
    init_lr = 0.00001
    max_lr = 0.001

    train_txt_path = "/data/Datasets/VOCdevkit/VOC2012/detection/train.txt"
    valid_txt_path = "/data/Datasets/VOCdevkit/VOC2012/detection/valid.txt"
    classes_path = "/data/Datasets/VOCdevkit/VOC2012/detection/Labels/labels.txt"
    save_name = "/data/Models/YoloX/weights.ckpt"

    class_names = read_label_file(classes_path)
    num_classes = len(class_names)

    train_files = read_txt_file(train_txt_path)
    valid_files = read_txt_file(valid_txt_path)
    num_train_files = len(train_files)
    num_valid_files = len(valid_files)

    train_dataloader = DataGenerator(train_files, input_shape, batch_size, num_classes, max_detections)
    valid_dataloader = DataGenerator(valid_files, input_shape, batch_size, num_classes, max_detections)

    optimizer = tf.keras.optimizers.Adam(learning_rate=init_lr)    
    clr = tfa.optimizers.CyclicalLearningRate(initial_learning_rate=init_lr,
                                              maximal_learning_rate=max_lr,
                                              scale_fn=lambda x : 1.0,
                                              step_size=epoch / 2)
    callbacks = [
        # DisplayCallback(),
        tf.keras.callbacks.LearningRateScheduler(clr),
        tf.keras.callbacks.ModelCheckpoint(save_name, save_best_only=True, save_weights_only=True, monitor="val_loss", verbose=1)
    ]

    with strategy.scope():
        model_body  = yolo_base([None, None, 3], num_classes = num_classes, phi = phi, weight_decay=weight_decay)
        loss = get_yolo_loss(input_shape, len(model_body.output), num_classes)

        model = get_train_model(model_body, input_shape, num_classes)
        model.compile(optimizer=optimizer, loss={"yolo_loss" : lambda y_true, y_pred : y_pred})
        model.summary()

    model.fit(train_dataloader,
              steps_per_epoch = num_train_files // batch_size,
              validation_data = valid_dataloader,
              validation_steps = num_valid_files // batch_size,
              epochs = epoch,
              callbacks = callbacks)