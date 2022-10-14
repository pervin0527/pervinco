import os
import cv2
import yaml
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from glob import glob
from IPython.display import clear_output
from data.dataloader import DataGenerator
from tensorflow.keras import backend as K
from models.centernet import centernet, get_train_model
from data.data_utils import read_label_file, read_txt_file


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
        
        if config["train"]["backbone"] == "resnet50":
            image = tf.keras.applications.resnet50.preprocess_input(image)

        elif config["train"]["backbone"] == "resnet101":
            image = tf.keras.applications.resnet50.preprocess_input(image)

        elif config["train"]["backbone"] == "mobilenet":
            image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
        image = np.expand_dims(image, axis=0)
        
        prediction = model.predict(image, verbose=0)
        bboxes, classes, scores = prediction[0], prediction[1], prediction[2]

        indices = np.where(scores[0] > config["train"]["threshold"])[0]
        if indices.size > 0:
            print(img_path, class_names[int(classes[0][indices][0])], scores[0][indices][0])


class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        plot_predictions(pred_model)


def cosine_decay_with_warmup(global_step,
                             learning_rate_base,
                             total_steps,
                             warmup_learning_rate=0.0,
                             warmup_steps=0,
                             hold_base_rate_steps=0):
    """Cosine decay schedule with warm up period.
    Cosine annealing learning rate as described in:
      Loshchilov and Hutter, SGDR: Stochastic Gradient Descent with Warm Restarts.
      ICLR 2017. https://arxiv.org/abs/1608.03983
    In this schedule, the learning rate grows linearly from warmup_learning_rate
    to learning_rate_base for warmup_steps, then transitions to a cosine decay
    schedule.
    Arguments:
        global_step {int} -- global step.
        learning_rate_base {float} -- base learning rate.
        total_steps {int} -- total number of training steps.
    Keyword Arguments:
        warmup_learning_rate {float} -- initial learning rate for warm up. (default: {0.0})
        warmup_steps {int} -- number of warmup steps. (default: {0})
        hold_base_rate_steps {int} -- Optional number of steps to hold base learning rate
                                    before decaying. (default: {0})
    Returns:
      a float representing learning rate.
    Raises:
      ValueError: if warmup_learning_rate is larger than learning_rate_base,
        or if warmup_steps is larger than total_steps.
    """

    if total_steps < warmup_steps:
        raise ValueError('total_steps must be larger or equal to warmup_steps.')
    learning_rate = 0.5 * learning_rate_base * (1 + np.cos(np.pi * (global_step - warmup_steps - hold_base_rate_steps) / float(total_steps - warmup_steps - hold_base_rate_steps)))
    if hold_base_rate_steps > 0:
        learning_rate = np.where(global_step > warmup_steps + hold_base_rate_steps, learning_rate, learning_rate_base)
    if warmup_steps > 0:
        if learning_rate_base < warmup_learning_rate:
            raise ValueError('learning_rate_base must be larger or equal to warmup_learning_rate.')
        slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
        warmup_rate = slope * global_step + warmup_learning_rate
        learning_rate = np.where(global_step < warmup_steps, warmup_rate, learning_rate)
    return np.where(global_step > total_steps, 0.0, learning_rate)


class WarmUpCosineDecayScheduler(tf.keras.callbacks.Callback):
    """Cosine decay with warmup learning rate scheduler"""
    def __init__(self,
                 learning_rate_base,
                 total_steps,
                 global_step_init=0,
                 warmup_learning_rate=0.0,
                 warmup_steps=0,
                 hold_base_rate_steps=0,
                 verbose=0):
        """Constructor for cosine decay with warmup learning rate scheduler.
    Arguments:
        learning_rate_base {float} -- base learning rate.
        total_steps {int} -- total number of training steps.
    Keyword Arguments:
        global_step_init {int} -- initial global step, e.g. from previous checkpoint.
        warmup_learning_rate {float} -- initial learning rate for warm up. (default: {0.0})
        warmup_steps {int} -- number of warmup steps. (default: {0})
        hold_base_rate_steps {int} -- Optional number of steps to hold base learning rate
                                    before decaying. (default: {0})
        verbose {int} -- 0: quiet, 1: update messages. (default: {0})
        """

        super(WarmUpCosineDecayScheduler, self).__init__()
        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.global_step = global_step_init
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.hold_base_rate_steps = hold_base_rate_steps
        self.verbose = verbose
        self.learning_rates = []

    def on_train_batch_end(self, batch, logs=None):
        self.global_step = self.global_step + 1
        lr = K.get_value(self.model.optimizer.lr)
        self.learning_rates.append(lr)

    def on_train_batch_begin(self, batch, logs=None):
        lr = cosine_decay_with_warmup(global_step=self.global_step,
                                      learning_rate_base=self.learning_rate_base,
                                      total_steps=self.total_steps,
                                      warmup_learning_rate=self.warmup_learning_rate,
                                      warmup_steps=self.warmup_steps,
                                      hold_base_rate_steps=self.hold_base_rate_steps)
        K.set_value(self.model.optimizer.lr, lr)
        # print('\n Batch %05d: setting learning rate to %s.' % (self.global_step, lr))

    def on_epoch_begin(self, epoch, logs=None):
        print(f'Epoch {epoch+1:>05} Start LR : {K.get_value(self.model.optimizer.lr)}')

    # def on_epoch_end(self, epoch, logs=None):
    #     print(f'Epoch {epoch:>05} End LR : {K.get_value(self.model.optimizer.lr)}')

# class LossAndLrPrintingCallback(tf.keras.callbacks.Callback):
#     def on_train_batch_end(self, batch, logs=None):
#         print(f'\r Train Step {batch} -  Loss : {logs["loss"]:.3f} -  LR : {K.get_value(self.model.optimizer.lr):.7f}', end="")

#     def on_test_batch_end(self, batch, logs=None):
#         print(f'\r Valid Step {batch} -  Loss : {logs["loss"]:.3f} -  LR : {K.get_value(self.model.optimizer.lr):.7f}', end="")



if __name__ == "__main__":
    with open("configs.yaml", "r") as f:
        config = yaml.safe_load(f)

    batch_size = config["train"]["batch_size"] * strategy.num_replicas_in_sync
    init_lr = config["train"]["init_lr"] * batch_size / 64
    print(f"Batch_size : {batch_size}, Init_lr : {init_lr}")

    class_names = read_label_file(config["path"]["label_path"])
    num_classes = len(class_names)

    if not os.path.isdir(config["path"]["save_path"]):
        os.makedirs(config["path"]["save_path"])

    train_lines = read_txt_file(config["path"]["train_txt_path"])
    valid_lines = read_txt_file(config["path"]["valid_txt_path"])

    train_steps = int(len(train_lines) // batch_size)
    validation_steps = int(len(valid_lines) // batch_size)

    train_dataloader = DataGenerator(train_lines, config["train"]["input_shape"], batch_size, num_classes, is_train = True, max_detections=config["train"]["max_detection"])
    val_dataloader = DataGenerator(valid_lines, config["train"]["input_shape"], batch_size, num_classes, is_train = False, max_detections=config["train"]["max_detection"])

    callbacks = [
        DisplayCallback(),
        #  LossAndLrPrintingCallback(),
        WarmUpCosineDecayScheduler(learning_rate_base=init_lr,
                                   total_steps=int(config["train"]["epoch"] * len(train_lines) / batch_size),
                                   warmup_learning_rate=config["train"]["warmup_lr"],
                                   warmup_steps=int(config["train"]["warmup_epoch"] * len(train_lines) / batch_size),
                                   hold_base_rate_steps=0),
        tf.keras.callbacks.ModelCheckpoint(config["path"]["save_path"] + '/' + config["path"]["ckpt_name"],
                                           save_best_only=True, 
                                           save_weights_only=True, 
                                           monitor="val_loss", 
                                           verbose=1),
        tf.keras.callbacks.TensorBoard(config["path"]["save_path"] + "/logs",
                                       write_graph=True,
                                       write_images=True, 
                                       write_steps_per_second=True, 
                                       update_freq="epoch")
    ]

    if config["train"]["optimizer"] == "adam":
        optimizer = tfa.optimizers.AdamW(weight_decay=config["train"]["weight_decay"], beta_1=0.9, beta_2=0.999)

    elif config["train"]["optimizer"] == "sgd":
        optimizer = tfa.optimizers.SGDW(weight_decay=config["train"]["weight_decay"], momentum=config["train"]["momentum"])

    with strategy.scope():
        base_model, pred_model = centernet(input_shape=[config["train"]["input_shape"][0], config["train"]["input_shape"][1], 3],
                                           num_classes=num_classes,
                                           backbone=config["train"]["backbone"],
                                           max_objects=config["train"]["max_detection"],
                                           weights=config["train"]["imagenet_weight"],
                                           mode="train")

        model = get_train_model(base_model=base_model,
                                input_shape=config["train"]["input_shape"],
                                num_classes=num_classes,
                                max_objects=config["train"]["max_detection"])

        if config["path"]["ckpt_path"]:
            model.load_weights(config["path"]["ckpt_path"], by_name=True, skip_mismatch=True)
            print("Weight Loaded")

        model.compile(optimizer = optimizer, loss = {'centernet_loss': lambda y_true, y_pred: y_pred})

    model.fit(train_dataloader,
              steps_per_epoch=train_steps,
              validation_data=val_dataloader,
              validation_steps=validation_steps,
              epochs=config["train"]["epoch"],
              callbacks=callbacks,
              verbose=1)