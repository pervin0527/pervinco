import os
import cv2
import yaml
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from glob import glob
from IPython.display import clear_output
from data.dataloader import DataGenerator
from models.centernet import centernet, get_train_model
from data.data_utils import read_label_file, read_txt_file, preprocess_input


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


def plot_predictions(model):
    for img_path in sorted(glob("./test_imgs/*.jpg")):
        image = cv2.imread(img_path)
        image = cv2.resize(image, config["train"]["input_shape"])
        # image = preprocess_input(image)
        image = np.expand_dims(image, axis=0)
        
        prediction = model.predict(image)[0]
        scores = prediction[:, 4]
        indices = np.where(scores > config["train"]["threshold"])

        if len(indices) > 0:
            print(prediction[indices])


class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        plot_predictions(pred_model)


if __name__ == "__main__":
    config_path = "./configs/train.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    class_names = read_label_file(config["path"]["label_path"])
    num_classes = len(class_names)

    if not os.path.isdir(config["path"]["save_path"]):
        os.makedirs(config["path"]["save_path"])

    if config["train"]["freeze_train"]:
        epoch = config["train"]["freeze_epoch"]
        batch_size = config["train"]["freeze_batch_size"]
        ckpt_name = "freeze.h5"
    else:
        epoch = config["train"]["unfreeze_epoch"]
        batch_size = config["train"]["unfreeze_batch_size"]
        ckpt_name = "unfreeze.h5"

    train_lines = read_txt_file(config["path"]["train_txt_path"])
    valid_lines = read_txt_file(config["path"]["valid_txt_path"])

    train_steps = int(len(train_lines) // batch_size)
    validation_steps = int(len(valid_lines) // batch_size)

    train_dataloader = DataGenerator(train_lines, config["train"]["input_shape"], batch_size, num_classes, is_train = True, max_detections=config["train"]["max_detection"])
    val_dataloader = DataGenerator(valid_lines, config["train"]["input_shape"], batch_size, num_classes, is_train = False, max_detections=config["train"]["max_detection"])

    clr = tfa.optimizers.CyclicalLearningRate(initial_learning_rate=config["train"]["init_lr"],
                                              maximal_learning_rate=config["train"]["max_lr"],
                                              scale_fn=lambda x : 1.0,
                                              step_size=epoch / 2)
    callbacks = [
        DisplayCallback(),
        tf.keras.callbacks.LearningRateScheduler(clr),
        tf.keras.callbacks.ModelCheckpoint(config["path"]["save_path"] + '/' + ckpt_name, save_best_only=True, save_weights_only=True, monitor="val_loss", verbose=1)
    ]

    if config["train"]["optimizer"] == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=config["train"]["init_lr"])
        # optimizer = tfa.optimizers.AdamW(learning_rate=config["train"]["init_lr"], weight_decay=config["train"]["weight_decay"], beta_1=0.9, beta_2=0.999)

    elif config["train"]["optimizer"] == "sgd":
        optimizer = tf.keras.optimizers.SGD(learning_rate=config["train"]["init_lr"])

    with strategy.scope():
        base_model, pred_model = centernet([config["train"]["input_shape"][0], config["train"]["input_shape"][1], 3],
                                            num_classes,
                                            backbone=config["train"]["backbone"],
                                            max_objects=config["train"]["max_detection"],
                                            mode="train")

        if config["train"]["freeze_train"]:
            if config["train"]["backbone"] == "resnet50":
                freeze_layers = 171
            elif config["train"]["backbone"] == "hourglass":
                freeze_layers = 624
            else:
                raise ValueError('Unsupported backbone - `{}`, Use resnet50, hourglass.'.format(config["train"]["backbone"]))
            for i in range(freeze_layers): base_model.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(freeze_layers, len(base_model.layers)))

        model = get_train_model(base_model, config["train"]["input_shape"], num_classes, config["train"]["backbone"])

        if config["path"]["model_path"]:
            model.load_weights(config["path"]["model_path"], by_name=True, skip_mismatch=True)
            print("Weight Loaded")

        model.compile(optimizer = optimizer, loss = {'centernet_loss': lambda y_true, y_pred: y_pred})

    model.fit(train_dataloader,
              steps_per_epoch = train_steps,
              validation_data = val_dataloader,
              validation_steps = validation_steps,
              epochs = epoch,
              callbacks=callbacks)