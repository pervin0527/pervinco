import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_addons as tfa

from model import DeepLabV3Plus
from glob import glob
from IPython.display import clear_output
from sklearn.model_selection import train_test_split

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


def get_file_list(path):
    images = sorted(glob(f"{path}/images/*.jpg"))
    masks = sorted(glob(f"{path}/masks/*.png"))
    
    n_images, n_masks = len(images), len(masks)
    
    return images, masks, n_images, n_masks


def read_image(image_path, mask=False):
    image = tf.io.read_file(image_path)

    if mask:
        image = tf.image.decode_png(image, channels=1)
        image.set_shape([None, None, 1])
        image = tf.image.resize(images=image, size=[IMG_SIZE, IMG_SIZE])

    else:
        image = tf.image.decode_png(image, channels=3)
        image.set_shape([None, None, 3])
        image = tf.image.resize(images=image, size=[IMG_SIZE, IMG_SIZE])

    return image


def load_data(image_list, mask_list):
    image = read_image(image_list)
    mask = read_image(mask_list, mask=True)

    return image, mask


def data_generator(image_list, mask_list):
    dataset = tf.data.Dataset.from_tensor_slices((image_list, mask_list))
    dataset = dataset.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.repeat()
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def infer(model, image_tensor):
    predictions = model.predict(np.expand_dims((image_tensor), axis=0))
    predictions = np.squeeze(predictions)
    predictions = np.argmax(predictions, axis=2)

    return predictions


def decode_segmentation_masks(mask, colormap, n_classes):
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    for l in range(0, n_classes):
        idx = mask == l
        r[idx] = colormap[l, 0]
        g[idx] = colormap[l, 1]
        b[idx] = colormap[l, 2]
    rgb = np.stack([r, g, b], axis=2)

    return rgb


def get_overlay(image, colored_mask):
    image = tf.keras.preprocessing.image.array_to_img(image)
    image = np.array(image).astype(np.uint8)
    overlay = cv2.addWeighted(image, 0.35, colored_mask, 0.65, 0)

    return overlay


def plot_samples_matplotlib(display_list, idx, figsize=(5, 3)):
    if not os.path.isdir("./train_result"):
        os.makedirs("./train_result")

    _, axes = plt.subplots(nrows=1, ncols=len(display_list), figsize=figsize)
    for i in range(len(display_list)):
        if display_list[i].shape[-1] == 3:
            axes[i].imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        else:
            axes[i].imshow(display_list[i])

    plt.savefig(f"./train_result/result_{idx}.png")
    # plt.show()
    plt.close()


def plot_predictions(images_list, colormap, model):
    for idx, image_file in enumerate(images_list):
        image_tensor = read_image(image_file)
        prediction_mask = infer(image_tensor=image_tensor, model=model)
        prediction_colormap = decode_segmentation_masks(prediction_mask, colormap, NUM_CLASSES)
        overlay = get_overlay(image_tensor, prediction_colormap)
        plot_samples_matplotlib([image_tensor, overlay, prediction_colormap], idx, figsize=(18, 14))


def display_training_curves(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(len(history.history['loss']))

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    
    plt.savefig(f"./train_result/train_history.png")
    # plt.show()
    plt.close()


def build_lrfn(lr_start=0.0001, lr_max=0.0005, lr_min=0.0001, lr_rampup_epochs=5, lr_sustain_epochs=0, lr_exp_decay=.8):
    lr_max = lr_max * strategy.num_replicas_in_sync

    def lrfn(epoch):
        if epoch < lr_rampup_epochs:
            lr = (lr_max - lr_start) / lr_rampup_epochs * epoch + lr_start
        elif epoch < lr_rampup_epochs + lr_sustain_epochs:
            lr = lr_max
        else:
            lr = (lr_max - lr_min) * lr_exp_decay**(epoch - lr_rampup_epochs - lr_sustain_epochs) + lr_min
        return lr
    return lrfn


class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        
        # idx = np.random.randint(len(valid_images))
        # plot_predictions([valid_images[idx]], colormap, model=model)
        
        plot_predictions(valid_images[:4], COLORMAP, model=model)


if __name__ == "__main__":
    ROOT = "/data/Datasets/VOCtrainval_11-May-2012/VOCdevkit/VOC2012"
    LABEL_PATH = f"{ROOT}/Labels/class_labels.txt"
    SAVE_PATH = "/data/Models/segmentation"    
    FOLDER = "SAMPLE01"

    BATCH_SIZE = 16
    EPOCHS = 300
    IMG_SIZE = 320
    LEARNING_RATE = 0.00001
    
    BACKBONE_TRAINABLE = True
    BACKBONE_NAME = "ResNet50"
    SAVE_NAME = f"{BACKBONE_NAME}_{EPOCHS}_trainable"

    label_df = pd.read_csv(LABEL_PATH, lineterminator='\n', header=None, index_col=False)
    CLASSES = label_df[0].to_list()
    NUM_CLASSES = len(CLASSES)
    print(CLASSES)

    COLORMAP = [[0, 0, 0],
                [128, 0, 0],
                [0, 128, 0],
                [128, 128, 0],
                [0, 0, 128],
                [128, 0, 128],
                [0, 128, 128],
                [128, 128, 128],
                [64, 0, 0],
                [192, 0, 0],
                [64, 128, 0],
                [192, 128, 0],
                [64, 0, 128],
                [192, 0, 128],
                [64, 128, 128],
                [192, 128, 128],
                [0, 64, 0],
                [128, 64, 0],
                [0, 192, 0],
                [128, 192, 0],
                [0, 64, 128]]
    COLORMAP = np.array(COLORMAP, dtype=np.uint8)


    root = f"{ROOT}/{FOLDER}"
    train_dir = f"{root}/train"
    valid_dir = f"{root}/valid"

    train_images, train_masks, n_train_images, n_train_masks = get_file_list(train_dir)
    valid_images, valid_masks, n_valid_images, n_valid_masks = get_file_list(valid_dir)

    train_dataset = data_generator(train_images, train_masks)
    valid_dataset = data_generator(valid_images, valid_masks)

    print("Train Dataset:", train_dataset)
    print("Val Dataset:", valid_dataset)

    model = DeepLabV3Plus(IMG_SIZE, IMG_SIZE, len(CLASSES), backbone_name=BACKBONE_NAME, backbone_trainable=BACKBONE_TRAINABLE)
    model.summary()

    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss=loss, metrics=["accuracy"])

    TRAIN_STEPS_PER_EPOCH = int(tf.math.ceil(len(train_images) / BATCH_SIZE).numpy())
    VALID_STEPS_PER_EPOCH = int(tf.math.ceil(len(valid_images) / BATCH_SIZE).numpy())

    lrfn = build_lrfn()
    callbacks = [DisplayCallback(),
                 tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=1),
                 tf.keras.callbacks.ModelCheckpoint(f"{SAVE_PATH}/{SAVE_NAME}/best.ckpt", monitor='val_loss', verbose=1, mode="min", save_best_only=True, save_weights_only=True)]
                
    history = model.fit(train_dataset,
                        steps_per_epoch=TRAIN_STEPS_PER_EPOCH,
                        validation_data=valid_dataset,
                        validation_steps=VALID_STEPS_PER_EPOCH,
                        callbacks=callbacks,
                        verbose=1,
                        epochs=EPOCHS)

    display_training_curves(history)
    plot_predictions(valid_images[:4], COLORMAP, model=model)

    run_model = tf.function(lambda x : model(x))
    BATCH_SIZE = 1
    concrete_func = run_model.get_concrete_function(tf.TensorSpec([BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3], model.inputs[0].dtype))
    tf.saved_model.save(model, f'{SAVE_PATH}/{SAVE_NAME}/saved_model', signatures=concrete_func)