import os
import cv2
import yaml
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from IPython.display import clear_output
from models import get_train_model, yolov3, DecodeBox
from loss import get_yolo_loss
from utils.utils import get_classes, preprocess_input
from utils.dataloader import DataGenerator

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
    trained_model = yolov3([config["input_shape"][0], config["input_shape"][1], 3], num_classes, config["phi"])
    trained_model.load_weights(config["save_path"] + '/weights.h5')

    outputs = tf.keras.layers.Lambda(DecodeBox, 
                                     output_shape = (1,), 
                                     name = 'yolo_eval',
                                     arguments = {'num_classes':num_classes, 
                                                  'input_shape':config["input_shape"], 
                                                  'confidence':config["score_threshold"], 
                                                  'nms_iou':config["iou_threshold"], 
                                                  'max_boxes':config["max_detections"], 
                                                  'letterbox_image':True})(trained_model.output)
    infer_model = tf.keras.Model(trained_model.input, outputs)

    image = cv2.imread("../centernet/test_imgs/1_0016.jpg")
    image = cv2.resize(image, config["input_shape"]).astype(np.float32)
    input_tensor = preprocess_input(image)
    input_tensor = np.expand_dims(input_tensor, axis=0)

    outputs = infer_model.predict(input_tensor)
    print(outputs)


class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        plot_predictions(model=model)


if __name__ == "__main__":
    with open("./config.yaml", "r") as f:
        config = yaml.safe_load(f)

    class_names, num_classes = get_classes(config["classes_path"])
    print(num_classes, class_names)

    with open(config["train_data_path"], encoding="utf-8") as f:
        train_lines = f.readlines()
    with open(config["valid_data_path"], encoding="utf-8") as f:
        valid_lines = f.readlines()

    num_train = len(train_lines)
    num_valid = len(valid_lines)
    print(num_train, num_valid)

    config["batch_size"] = config["batch_size"] * strategy.num_replicas_in_sync
    train_dataloader = DataGenerator(train_lines, config["input_shape"], config["batch_size"], num_classes, config["max_detections"])
    valid_dataloader = DataGenerator(valid_lines, config["input_shape"], config["batch_size"], num_classes, config["max_detections"])

    for data in train_dataloader:
        print(data[0][0].shape, data[0][1].shape)
        break

    clr = tfa.optimizers.CyclicalLearningRate(initial_learning_rate=config["init_lr"],
                                              maximal_learning_rate=config["max_lr"],
                                              scale_fn=lambda x : 1.0,
                                              step_size=config["epochs"] / 2)

    callbacks = [
        # DisplayCallback(),
        tf.keras.callbacks.LearningRateScheduler(clr),
        tf.keras.callbacks.ModelCheckpoint(config["save_path"] + "/weights.h5",
                                           save_best_only=True,
                                           save_weights_onlyt=True,
                                           monitor="val_loss",
                                           verbose=1)
    ]

    with strategy.scope():
        base_model = yolov3([config["input_shape"][0], config["input_shape"][1], 3],
                                num_classes=num_classes,
                                phi=config["phi"],
                                weight_decay=config["weight_decay"])
        
        model = get_train_model(base_model, config["input_shape"], num_classes)
        if config["ckpt_path"]:
            model.load_weights(config["ckpt_path"])

        # optimizer = tfa.optimizers.AdamW(learning_rate=config["init_lr"], weight_decay=config["weight_decay"], beta_1=0.9, beta_2=0.999)
        optimizer = tf.keras.optimizers.Adam(learning_rate=config["init_lr"])
        loss = get_yolo_loss(config["input_shape"], len(base_model.output), num_classes)
        model.compile(optimizer=optimizer, loss={'yolo_loss': lambda y_true, y_pred: y_pred})

    model.fit(train_dataloader,
              steps_per_epoch=num_train / config["batch_size"],
              validation_data=valid_dataloader,
              validation_steps=num_valid / config["batch_size"],
              epochs=config["epochs"],
              callbacks=callbacks)