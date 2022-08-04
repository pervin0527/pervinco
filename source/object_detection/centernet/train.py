import os
import numpy as np
import tensorflow as tf
from glob import glob
from model import CenterNet
from data_loader import DataGenerator
from IPython.display import clear_output

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
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
        resized = tf.image.resize(image, (input_shape[0], input_shape[1]))
        input_tensor = tf.expand_dims(resized, axis=0)

        prediction = model.predict(input_tensor, verbose=0)[-1][0]
        scores = prediction[:, 4]
        indexes = np.where(scores > 0.7)
        detections = prediction[indexes]
        print(detections)

        break


class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        plot_predictions(model=model)


if __name__ == "__main__":
    epochs = 500
    batch_size = 32
    classes = ["face"]
    max_detections = 10
    backbone = "resnet18"
    learning_rate = 0.001
    input_shape = (512, 512, 3)
    save_dir = "/data/Models/FACE_DETECTION/CenterNet"
    
    train_txt = "/data/Datasets/WIDER/FACE/train_512/annot.txt"
    test_txt = "/data/Datasets/WIDER/FACE/test_512/annot.txt"

    train_dataset = DataGenerator(train_txt, classes, batch_size, (input_shape[0], input_shape[1]), max_detections)
    train_steps = int(tf.math.ceil(len(train_dataset) / batch_size).numpy())

    test_dataset = DataGenerator(test_txt, classes, batch_size, (input_shape[0], input_shape[1]), max_detections)
    test_steps = int(tf.math.ceil(len(test_dataset) / batch_size).numpy())

    optimizer = tf.keras.optimizers.Adam()
    cdr = tf.keras.optimizers.schedules.CosineDecayRestarts(initial_learning_rate=learning_rate,
                                                            first_decay_steps=200,
                                                            t_mul=2.0,
                                                            m_mul=0.8,
                                                            alpha=learning_rate * 0.01)

    callbacks = [
        DisplayCallback(),
        tf.keras.callbacks.LearningRateScheduler(cdr),
        # tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5, verbose=1, mode="min", factor=0.9, min_delta=0.01, min_lr=1e-5),
        tf.keras.callbacks.TensorBoard(log_dir=f"{save_dir}/TensorBoard", update_freq='epoch'),
        tf.keras.callbacks.ModelCheckpoint(f"{save_dir}/ckpt.h5", monitor="val_loss", verbose=1, save_best_only=True, save_weights_only=True)
    ]

    with strategy.scope():
        model = CenterNet(inputs=input_shape, num_classes=len(classes), max_detections=max_detections, backbone=backbone)
        model.trainable=True
        model.compile(optimizer=optimizer)
    
    model.fit(train_dataset,
              steps_per_epoch=train_steps,
              epochs = epochs,
              validation_data=test_dataset,
              validation_steps=test_steps,
              callbacks=callbacks)