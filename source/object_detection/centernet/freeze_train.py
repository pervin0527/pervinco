import os
import numpy as np
import tensorflow as tf
from glob import glob
from model import CenterNet
from data_loader import DataGenerator
from IPython.display import clear_output

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
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
    for idx, file in enumerate(sorted(glob("./samples/*"))):
        file = tf.io.read_file(file)
        image = tf.io.decode_jpeg(file, channels=3)
        resized = tf.image.resize(image, (input_shape[0], input_shape[1])) / 255.0
        input_tensor = tf.expand_dims(resized, axis=0)

        prediction = pred_model.predict(input_tensor, verbose=0)[0]
        scores = prediction[:, 4]
        indexes = np.where(scores > 0.6)[0]
        detections = prediction[indexes]
        print(detections)

        break


class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        plot_predictions(model=model)


if __name__ == "__main__":
    batch_size = 32
    classes = ["face"]
    max_detections = 10
    input_shape = (512, 512, 3)

    freeze_backbone = True
    backbone = "resnet18"
    train_dir = "/home/ubuntu/Datasets/WIDER/FACE/train_512"
    test_dir = "/home/ubuntu/Datasets/WIDER/FACE/test_512"
    save_dir = "/home/ubuntu/Models/FACE_DETECTION/CenterNet-test"

    train_dataset = DataGenerator(train_dir, classes, batch_size, (input_shape[0], input_shape[1]), max_detections)
    train_steps = int(tf.math.ceil(len(train_dataset) / batch_size).numpy())

    test_dataset = DataGenerator(test_dir, classes, batch_size, (input_shape[0], input_shape[1]), max_detections)
    test_steps = int(tf.math.ceil(len(test_dataset) / batch_size).numpy())

    if freeze_backbone:
        epochs = 200
        learning_rate = 0.0001
        ckpt_name = "freezed.h5"

    else:
        epochs = 500
        learning_rate = 0.001
        ckpt_name = "unfreezed.h5"

    callbacks = [
        DisplayCallback(),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5, verbose=1, mode="min", factor=0.9, min_delta=0.01, min_lr=1e-5),
        tf.keras.callbacks.TensorBoard(log_dir=f"{save_dir}/TensorBoard", update_freq='epoch'),
        tf.keras.callbacks.ModelCheckpoint(f"{save_dir}/{ckpt_name}", monitor="val_loss", verbose=1, save_best_only=True, save_weights_only=True)
    ]

    optimizer = tf.keras.optimizers.Adam()
    with strategy.scope():
        model, pred_model = CenterNet(input_shape, len(classes), max_detections, backbone)
        model.compile(optimizer=optimizer, loss={'centernet_loss': lambda y_true, y_pred: y_pred})

        if freeze_backbone:
            for i in range(85): # resnet18 : 85
                model.layers[i].trainable = False

        else:
            model.load_weights("/home/ubuntu/Models/FACE_DETECTION/CenterNet-test/freezed.h5", by_name=True, skip_mismatch=True)
    
    model.fit(train_dataset,
              steps_per_epoch=train_steps,
              epochs = epochs,
              validation_data=test_dataset,
              validation_steps=test_steps,
              callbacks=callbacks)