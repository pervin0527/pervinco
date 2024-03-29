import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from glob import glob
from model import DeepLabV3Plus
from calculate_class_weights import analyze_dataset, create_class_weight
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


def visualize(display_list):
    if not os.path.isdir("./images/train"):
        os.makedirs("./images/train")

    fig = plt.figure(figsize=(8, 5))
    rows, cols = 1, 2

    x_labels = ["Train image", "Train mask"]
    for idx, image in enumerate(display_list):
        ax = fig.add_subplot(rows, cols, idx+1)
        # if image.shape[-1] == 3:
        #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax.imshow(image)
        ax.set_xlabel(x_labels[idx])
        ax.set_xticks([]), ax.set_yticks([])
    
    plt.savefig(f"./images/train/batch_sample_{idx}.png")
    # plt.show()
    plt.close()


def get_file_list(path):
    images = sorted(glob(f"{path}/images/*.jpg"))
    masks = sorted(glob(f"{path}/masks/*.png"))
    
    n_images, n_masks = len(images), len(masks)
    
    return images, masks, n_images, n_masks


def read_image(image_path, mask=False):
    image = tf.io.read_file(image_path)

    if mask:
        image = tf.image.decode_png(image, channels=1)
        image = tf.image.resize(images=image, size=[IMG_SIZE, IMG_SIZE])
        image.set_shape([IMG_SIZE, IMG_SIZE, 1])

        if CATEGORICAL:
            image = tf.cast(image, tf.uint8)
            image = tf.squeeze(image, axis=-1)
            image = tf.one_hot(image, len(CLASSES))

    else:
        image = tf.image.decode_png(image, channels=3)
        image = tf.image.resize(images=image, size=[IMG_SIZE, IMG_SIZE])
        image.set_shape([IMG_SIZE, IMG_SIZE, 3])
        image = image / 127.5 - 1

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
    if not os.path.isdir("./images/train"):
        os.makedirs("./images/train")

    _, axes = plt.subplots(nrows=1, ncols=len(display_list), figsize=figsize)
    for i in range(len(display_list)):
        if display_list[i].shape[-1] == 3:
            axes[i].imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        else:
            axes[i].imshow(display_list[i])

    plt.savefig(f"./images/train/train_result_{idx}.png")
    # plt.show()
    plt.close()


def plot_predictions(images_list, colormap, model):
    for idx, image_file in enumerate(images_list):
        image_tensor = read_image(image_file)
        prediction_mask = infer(image_tensor=image_tensor, model=model)
        prediction_colormap = decode_segmentation_masks(prediction_mask, colormap, len(CLASSES))
        overlay = get_overlay(image_tensor, prediction_colormap)
        plot_samples_matplotlib([image_tensor, overlay, prediction_colormap], idx, figsize=(14, 12))


class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        
        plot_predictions(valid_images[:4], COLORMAP, model=model)


class Sparse_MeanIoU(tf.keras.metrics.MeanIoU):
  def __init__(self,
               y_true=None,
               y_pred=None,
               num_classes=None,
               name=None,
               dtype=None):
    super(Sparse_MeanIoU, self).__init__(num_classes = num_classes,name=name, dtype=dtype)

  def update_state(self, y_true, y_pred, sample_weight=None):
    y_pred = tf.math.argmax(y_pred, axis=-1)
    return super().update_state(y_true, y_pred, sample_weight)


def get_model():
    with strategy.scope():
    
        if CATEGORICAL:
            # dice_loss = advisor.losses.DiceLoss(class_weights=CLASS_WEIGHTS)
            # categorical_focal_loss = advisor.losses.CategoricalFocalLoss()
            # loss = dice_loss + (1 * categorical_focal_loss)
            loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
            metrics = tf.keras.metrics.OneHotMeanIoU(num_classes=len(CLASSES))

        else:
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            # metrics = "accuracy"
            metrics = Sparse_MeanIoU(num_classes=len(CLASSES))

        optimizer = tf.keras.optimizers.Adam(learning_rate=LR_START)
        model = DeepLabV3Plus(IMG_SIZE, IMG_SIZE, len(CLASSES), backbone_name=BACKBONE_NAME, backbone_trainable=BACKBONE_TRAINABLE, final_activation=FINAL_ACTIVATION)

        model.summary()
        model.compile(optimizer=optimizer, loss=loss, metrics=[metrics])

    return model


if __name__ == "__main__":
    ROOT = "/home/ubuntu/Datasets/VOCdevkit/VOC2012"
    LABEL_PATH = f"{ROOT}/Labels/class_labels.txt"
    SAVE_PATH = "/home/ubuntu/Models/segmentation"    
    FOLDER = "AUGMENT_50"

    VIS_SAMPLE = False
    CATEGORICAL = False
    BACKBONE_TRAINABLE = True
    BACKBONE_NAME = "ResNet101" # Xception, ResNet50, ResNet101
    FINAL_ACTIVATION = None # None, softmax
    SAVE_NAME = f"{ROOT.split('/')[-1]}-{BACKBONE_NAME}-{FOLDER}"
    CKPT_PATH = f"{SAVE_PATH}/VOC2012-ResNet101-AUGMENT_50/best.ckpt"

    BATCH_SIZE = 64
    EPOCHS = 300
    IMG_SIZE = 320
    ES_PATIENT = 10
    
    LR_START = 0.0001
    LR_MAX = 0.0005
    LR_MIN = 0.0001
    LR_RAMPUP_EPOCHS = 4
    LR_SUSTAIN_EPOCHS = 4
    LR_EXP_DECAY = .8

    label_df = pd.read_csv(LABEL_PATH, lineterminator='\n', header=None, index_col=False)
    CLASSES = label_df[0].to_list()
    print(CLASSES)

    COLORMAP = [[0, 0, 0], # background
                [128, 0, 0], # aeroplane
                [0, 128, 0], # bicycle
                [128, 128, 0], # bird
                [0, 0, 128], # boat
                [128, 0, 128], # bottle
                [0, 128, 128], # bus
                [128, 128, 128], # car
                [64, 0, 0], # cat
                [192, 0, 0], # chair
                [64, 128, 0], # cow
                [192, 128, 0], # diningtable
                [64, 0, 128], # dog
                [192, 0, 128], # horse
                [64, 128, 128], # motorbike
                [192, 128, 128], # person
                [0, 64, 0], # potted plant
                [128, 64, 0], # sheep
                [0, 192, 0], # sofa
                [128, 192, 0], # train
                [0, 64, 128] # tv/monitor
    ]
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

    # CLASS_PER_PIXEL = analyze_dataset(train_masks, CLASSES, IMG_SIZE, IMG_SIZE)
    # CLASS_WEIGHTS = create_class_weight(CLASS_PER_PIXEL)
    # CLASS_WEIGHTS = list(CLASS_WEIGHTS.values())
    # CLASS_WEIGHTS = np.array(CLASS_WEIGHTS)

    if VIS_SAMPLE:
        for item in train_dataset.take(4):
            image, mask = item[0][0], item[1][0]
            image = image.numpy()
            mask = decode_segmentation_masks(mask.numpy(), COLORMAP, len(CLASSES))
            mask = np.squeeze(mask, axis=-1)

            visualize([image, mask])
    
    TRAIN_STEPS_PER_EPOCH = int(tf.math.ceil(len(train_images) / BATCH_SIZE).numpy())
    VALID_STEPS_PER_EPOCH = int(tf.math.ceil(len(valid_images) / BATCH_SIZE).numpy())

    callbacks = [DisplayCallback(),
                #  tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True),
                #  tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=ES_PATIENT, verbose=1),
                #  tf.keras.callbacks.ModelCheckpoint(f"{SAVE_PATH}/{SAVE_NAME}/best.ckpt", monitor='val_one_hot_mean_io_u', verbose=1, mode="max", save_best_only=True, save_weights_only=True)
                #  tf.keras.callbacks.ModelCheckpoint(f"{SAVE_PATH}/{SAVE_NAME}/best.ckpt", monitor='val_loss', verbose=1, mode="min", save_best_only=True, save_weights_only=True)
                 tf.keras.callbacks.ModelCheckpoint(f"{SAVE_PATH}/{SAVE_NAME}/best.ckpt", monitor='val_sparse__mean_io_u', verbose=1, mode="max", save_best_only=True, save_weights_only=True)
                 ]

    model = get_model()
    if CKPT_PATH != None:
        model.load_weights(CKPT_PATH)
    history = model.fit(train_dataset,
                        steps_per_epoch=TRAIN_STEPS_PER_EPOCH,
                        validation_data=valid_dataset,
                        validation_steps=VALID_STEPS_PER_EPOCH,
                        callbacks=callbacks,
                        verbose=1,
                        epochs=EPOCHS)

    plot_predictions(valid_images[:4], COLORMAP, model=model)

    run_model = tf.function(lambda x : model(x))
    BATCH_SIZE = 1
    concrete_func = run_model.get_concrete_function(tf.TensorSpec([BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3], model.inputs[0].dtype))
    tf.saved_model.save(model, f'{SAVE_PATH}/{SAVE_NAME}/saved_model', signatures=concrete_func)