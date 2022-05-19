import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from model import DeepLabV3Plus
from utils import plot_predictions
from metrics import Sparse_MeanIoU
from hparams_config import send_params, save_params
from data import get_file_list, data_generator
from IPython.display import clear_output

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


def build_model(checkpoint):
    with strategy.scope():
        if params["ONE_HOT"]:
            loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
            metrics = tf.keras.losses.OneHotMeanIoU(num_classes=len(params["CLASSES"]))
        else:
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            metrics = Sparse_MeanIoU(num_classes=len(params["CLASSES"]))

        optimizer = tf.keras.optimizers.Adam(learning_rate=params["LR_START"])
        
        model = DeepLabV3Plus(params["IMG_SIZE"], params["IMG_SIZE"], len(params["CLASSES"]), backbone_name=params["BACKBONE_NAME"], backbone_trainable=params["BACKBONE_TRAINABLE"], final_activation=params["FINAL_ACTIVATION"])
        model.compile(optimizer=optimizer, loss=loss, metrics=[metrics])
        model.summary()

    if checkpoint != None:
        model.load_weights(checkpoint)

    return model


if __name__ == "__main__":
    params = send_params(show_contents=False)
    save_ckpt = params["SAVE_PATH"]
    callbacks = [
        DisplayCallback(),
        tf.keras.callbacks.ModelCheckpoint(f"{save_ckpt}/best,ckpt", monitor="val_sparse__mean_io_u", verbose=1, mode="max", save_best_only=True, save_weights_only=True)
    ]

    data_dir = params["DATASET_PATH"]
    train_dir = f"{data_dir}/train"
    valid_dir = f"{data_dir}/valid"

    train_images, train_masks, n_train_images, n_train_masks = get_file_list(train_dir)
    valid_images, valid_masks, n_valid_images, n_valid_masks = get_file_list(valid_dir)

    train_dataset = data_generator(train_images, train_masks, params["BATCH_SIZE"])
    valid_dataset = data_generator(valid_images, valid_masks, params["BATCH_SIZE"])

    print("Train Dataset : ", train_dataset)
    print("Valid Dataset : ", train_dataset)

    TRAIN_STEPS_PER_EPOCH = int(tf.math.ceil(len(train_images) / params["BATCH_SIZE"]).numpy())
    VALID_STEPS_PER_EPOCH = int(tf.math.ceil(len(valid_images) / params["BATCH_SIZE"]).numpy())

    model = build_model(params["CKPT"])
    history = model.fit(
        train_dataset,
        steps_per_epoch=TRAIN_STEPS_PER_EPOCH,
        validation_data=valid_dataset,
        validation_steps=VALID_STEPS_PER_EPOCH,
        callbacks=callbacks,
        verbose=1,
        epochs=params["EPOCHS"]
    )

    plot_predictions(valid_images[:5], params["COLORMAP"], model=model)
    save_params()