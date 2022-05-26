import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf

from IPython.display import clear_output
from advisor import losses
from model import DeepLabV3Plus
from utils import plot_predictions
from metrics import Sparse_MeanIoU
from data import get_file_list, data_generator
from hparams_config import send_params, save_params
from calculate_class_weights import analyze_dataset, create_class_weight
from loss import SparseCategoricalFocalLoss, categorical_focal_loss, dice_coefficient_loss

# GPU setup
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
        
        plot_predictions(valid_images[:5], params["COLORMAP"], model=model)


def build_model():
    with strategy.scope():
        if params["ONE_HOT"]:
            # loss = dice_coefficient_loss(smooth=1)
            # loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
            loss = categorical_focal_loss(gamma=2, alpha=0.25)
            metrics = tf.keras.metrics.OneHotMeanIoU(num_classes=len(params["CLASSES"]))

        else:
            # loss = SparseCategoricalFocalLoss(gamma=2, class_weight=class_weights, from_logits=True)
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            metrics = Sparse_MeanIoU(num_classes=len(params["CLASSES"]))

        optimizer = tf.keras.optimizers.Adam(learning_rate=params["LR_START"])
        
        model = DeepLabV3Plus(params["IMG_SIZE"], params["IMG_SIZE"],
                              len(params["CLASSES"]),
                              backbone_name=params["BACKBONE_NAME"],
                              backbone_trainable=params["BACKBONE_TRAINABLE"],
                              final_activation=params["FINAL_ACTIVATION"],
                              original_output=params["ORIGINAL_OUTPUT"])
        model.compile(optimizer=optimizer, loss=loss, metrics=[metrics])
        model.summary()

    return model


if __name__ == "__main__":
    params = send_params(show_contents=False)
    save_ckpt = params["SAVE_PATH"]
    one_hot = params["ONE_HOT"]
    monitor = None
    if one_hot: 
        monitor = "val_one_hot_mean_io_u"
    else: 
        monitor = "val_sparse__mean_io_u"

    callbacks = [
        DisplayCallback(),
        tf.keras.callbacks.ModelCheckpoint(f"{save_ckpt}/best.ckpt", monitor=monitor,
                                           verbose=1, mode="max", save_best_only=True, save_weights_only=True),
        tf.keras.callbacks.TensorBoard(log_dir=f"{save_ckpt}/logs", write_graph=True, write_images=True, update_freq="epoch")                                           
    ]

    data_dir = params["DATASET_PATH"]
    train_dir = f"{data_dir}/train"
    valid_dir = f"{data_dir}/valid"

    total_images, total_masks, n_total_images, n_total_masks = get_file_list(train_dir)
    valid_images, valid_masks, n_valid_images, n_valid_masks = get_file_list(valid_dir)

    for number, k in enumerate(range(7)):
        start = k * 20000
        end = start + 20000

        if number == 6:
            end = n_total_images

        train_images, train_masks = total_images[start:end], total_masks[start:end]
        if params["INCLUDE_CLASS_WEIGHT"]:
            class_per_pixels = analyze_dataset(train_masks, params["CLASSES"], height=params["IMG_SIZE"], width=params["IMG_SIZE"])
            class_weights = create_class_weight(class_per_pixels)
            class_weights = list(class_weights.values())

        train_dataset = data_generator(train_images, train_masks, params["BATCH_SIZE"])
        valid_dataset = data_generator(valid_images, valid_masks, params["BATCH_SIZE"])

        print("Train Dataset:", train_dataset)
        print("Val Dataset:", valid_dataset)
        
        TRAIN_STEPS_PER_EPOCH = int(tf.math.ceil(len(train_images) / params["BATCH_SIZE"]).numpy())
        VALID_STEPS_PER_EPOCH = int(tf.math.ceil(len(valid_images) / params["BATCH_SIZE"]).numpy())

        model = build_model()
        if number != 0:
            ckpt_path = params["SAVE_PATH"]
            model.load_weights(f"{ckpt_path}/best.ckpt")

        print(f"FOLD_{number} STARTING")
        history = model.fit(train_dataset,
                            steps_per_epoch=TRAIN_STEPS_PER_EPOCH,
                            validation_data=valid_dataset,
                            validation_steps=VALID_STEPS_PER_EPOCH,
                            callbacks=callbacks,
                            verbose=1,
                            epochs=params["EPOCHS"])

        print(f"FOLD {number} END \n")
        tf.keras.backend.clear_session()

    print("Train Finished \n")
    plot_predictions(valid_images[:5], params["COLORMAP"], model=model)