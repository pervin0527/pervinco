import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import advisor
import numpy as np
import pandas as pd
import tensorflow as tf
import albumentations as A
import matplotlib.pyplot as plt

from generator import TFDataGenerator
from glob import glob
from augwrap.src import nightly as aw
from IPython.display import clear_output
from augwrap.src.nightly.augmentations import CutMix
from class_weight_helper import get_balancing_class_weights


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


def get_training_augmentation(height, width, dataset):
    train_transform = [
        A.Resize(height, width, always_apply=True),
        A.OneOf([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            # A.Transpose(p=0.5),
        ], p=0.7),

        A.OneOf([
            A.GridDropout(fill_value=0, mask_fill_value=0, random_offset=True, p=0.5),
            A.CoarseDropout(min_holes=1, max_holes=1,
                            min_height=80, min_width=80,
                            max_height=160, max_width=160,
                            fill_value=0, mask_fill_value=0,
                            p=0.5,),
        ], p=0.5), 

        A.ShiftScaleRotate(scale_limit=0.6, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
    #    A.PadIfNeeded(min_height=height, min_width=width, always_apply=True, border_mode=0),
    #    A.RandomCrop(height=height, width=width, always_apply=True),

        A.GaussNoise(p=0.2),
        A.Perspective(p=0.5),

        CutMix(dataset, p=0.5),

        A.OneOf(
            [
                A.CLAHE(p=1),
                A.RandomBrightnessContrast(p=1),
                A.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        A.OneOf(
            [
                A.Sharpen(p=1),
                A.Blur(blur_limit=3, p=1),
                A.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        A.OneOf(
            [
                A.RandomBrightnessContrast(p=1),
                A.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return A.Compose(train_transform)


def get_validation_augmentation(height, width):
    test_transform = [
        # A.PadIfNeeded(height, width),
        A.Resize(height, width, always_apply=True)
    ]
    return A.Compose(test_transform)


def data_get_preprocessing(preprocessing_fn):
    _transform = [A.Lambda(image=preprocessing_fn),]
    return A.Compose(_transform)


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
    for idx, image in enumerate(images_list):
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        prediction_mask = infer(image_tensor=image, model=model)
        prediction_colormap = decode_segmentation_masks(prediction_mask, colormap, len(CLASSES))
        overlay = get_overlay(image, prediction_colormap)
        plot_samples_matplotlib([image, overlay, prediction_colormap], idx, figsize=(14, 12))


class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        
        # idx = np.random.randint(len(valid_images))
        # plot_predictions([valid_images[idx]], colormap, model=model)
        plot_predictions(valid_images, COLORMAP, model=model)


def AtrousSpatialPyramidPooling(model_input):
    dims = tf.keras.backend.int_shape(model_input)

    layer = tf.keras.layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(model_input)
    layer = tf.keras.layers.Conv2D(256, kernel_size=1, padding='same', kernel_initializer = 'he_normal')(layer)
    layer = tf.keras.layers.BatchNormalization()(layer)
    layer = tf.keras.layers.ReLU()(layer)
    out_pool = tf.keras.layers.UpSampling2D(size = (dims[-3] // layer.shape[1], dims[-2] // layer.shape[2]), interpolation = 'bilinear')(layer)
  
    layer = tf.keras.layers.Conv2D(256, kernel_size = 1, dilation_rate = 1, padding = 'same', kernel_initializer = 'he_normal', use_bias = False)(model_input)
    layer = tf.keras.layers.BatchNormalization()(layer)
    out_1 = tf.keras.layers.ReLU()(layer)

    layer = tf.keras.layers.Conv2D(256, kernel_size = 3, dilation_rate = 6, padding = 'same', kernel_initializer = 'he_normal', use_bias = False)(model_input)
    layer = tf.keras.layers.BatchNormalization()(layer)
    out_6 = tf.keras.layers.ReLU()(layer)

    layer = tf.keras.layers.Conv2D(256, kernel_size = 3, dilation_rate = 12, padding = 'same', kernel_initializer = 'he_normal', use_bias = False)(model_input)
    layer = tf.keras.layers.BatchNormalization()(layer)
    out_12 = tf.keras.layers.ReLU()(layer)

    layer = tf.keras.layers.Conv2D(256, kernel_size = 3, dilation_rate = 18, padding = 'same', kernel_initializer = 'he_normal', use_bias = False)(model_input)
    layer = tf.keras.layers.BatchNormalization()(layer)
    out_18 = tf.keras.layers.ReLU()(layer)

    layer = tf.keras.layers.Concatenate(axis = -1)([out_pool, out_1, out_6, out_12, out_18])

    layer = tf.keras.layers.Conv2D(256, kernel_size = 1,  dilation_rate = 1, padding = 'same', kernel_initializer = 'he_normal', use_bias = False)(layer)
    layer = tf.keras.layers.BatchNormalization()(layer)
    model_output = tf.keras.layers.ReLU()(layer)
    return model_output


def DeeplabV3Plus(nclasses = 20):
    # layer_name = ['block7b_expand_activation', 'block3a_expand_activation']
    layer_name = ['block7b_expand_activation', 'block2a_activation']

    model_input = tf.keras.Input(shape=(HEIGHT,WIDTH,3))
    model_input = tf.keras.layers.experimental.preprocessing.Rescaling(1.0 / 255.0)(model_input)
    
    effnet = tf.keras.applications.EfficientNetB3(weights = 'imagenet', include_top = False, input_tensor = model_input)
    layer = effnet.get_layer(layer_name[0]).output
    layer = AtrousSpatialPyramidPooling(layer)
    input_a = tf.keras.layers.UpSampling2D(size = (HEIGHT // 4 // layer.shape[1], WIDTH // 4 // layer.shape[2]), interpolation = 'bilinear')(layer)

    input_b = effnet.get_layer(layer_name[1]).output
    input_b = tf.keras.layers.Conv2D(48, kernel_size = (1,1), padding = 'same', kernel_initializer = tf.keras.initializers.he_normal(), use_bias = False)(input_b)
    input_b = tf.keras.layers.BatchNormalization()(input_b)
    input_b = tf.keras.layers.ReLU()(input_b)

    print(input_a.shape, input_b.shape)
    layer = tf.keras.layers.Concatenate(axis = -1)([input_a, input_b])

    layer = tf.keras.layers.Conv2D(256, kernel_size = 3, padding = 'same', activation = 'relu', kernel_initializer = tf.keras.initializers.he_normal(), use_bias = False)(layer)
    layer = tf.keras.layers.BatchNormalization()(layer)
    layer = tf.keras.layers.ReLU()(layer)
    layer = tf.keras.layers.Conv2D(256, kernel_size =3, padding = 'same', activation = 'relu', kernel_initializer = tf.keras.initializers.he_normal(), use_bias = False)(layer)
    layer = tf.keras.layers.BatchNormalization()(layer)
    layer = tf.keras.layers.ReLU()(layer)
    layer = tf.keras.layers.UpSampling2D(size = (HEIGHT // layer.shape[1], WIDTH // layer.shape[2]), interpolation = 'bilinear')(layer)
    model_output = tf.keras.layers.Conv2D(len(CLASSES), kernel_size = (1,1), padding = 'same')(layer)
    model_output = tf.keras.layers.Activation("softmax")(model_output)
    return tf.keras.Model(inputs = model_input, outputs = model_output)


def get_model():
    with strategy.scope(): 
        # dice_loss = advisor.losses.DiceLoss(class_weights=None)
        # categorical_focal_loss = advisor.losses.CategoricalFocalLoss()
        # loss = dice_loss + (1 * categorical_focal_loss)

        # metrics = [advisor.metrics.FScore(threshold=0.5), advisor.metrics.IOUScore(threshold=0.5)]
        # metrics = [advisor.metrics.IOUScore()]

        # loss = tf.losses.SparseCategoricalCrossentropy(from_logits=False)
        loss  = tf.keras.losses.CategoricalCrossentropy()
        metrics = tf.keras.metrics.CategoricalAccuracy()

        optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
        model = DeeplabV3Plus(len(CLASSES))

        # model.summary()
        model.compile(optimizer=optimizer, loss=loss, metrics=[metrics])

    return model


if __name__ == "__main__":
    ROOT = "/data/Datasets/VOCdevkit/VOC2012"
    LABEL_PATH = f"{ROOT}/Labels/class_labels.txt"
    SAVE_PATH = "/data/Models/segmentation"
    FOLDER = "BASIC"

    x_train_dir, y_train_dir = f"{ROOT}/{FOLDER}/train/images", f"{ROOT}/{FOLDER}/train/masks"
    x_valid_dir, y_valid_dir = f"{ROOT}/{FOLDER}/valid/images", f"{ROOT}/{FOLDER}/valid/masks"

    LR = 0.001
    EPOCHS = 3000
    BATCH_SIZE = 16
    ES_PATIENT = 10
    HEIGHT, WIDTH = 320, 320
    BACKBONE_NAME = "ResNet101"
    BACKBONE_TRAINABLE = True
    FINAL_ACTIVATION = "softmax"
    SAVE_NAME = f"{ROOT.split('/')[-1]}-{BACKBONE_NAME}-{FOLDER}-{EPOCHS}"

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

    CLASSES_PIXEL_COUNT_DICT = {'background': 361560627, 'aeroplane': 3704393, 'bicycle': 1571148, 'bird': 4384132,
                                'boat': 2862913, 'bottle': 3438963, 'bus': 8696374, 'car': 7088203, 'cat': 12473466,
                                'chair': 4975284, 'cow': 5027769, 'diningtable': 6246382, 'dog': 9379340, 'horse': 4925676,
                                'motorbike': 5476081, 'person': 24995476, 'potted plant': 2904902, 'sheep': 4187268, 'sofa': 7091464, 'train': 7903243, 'tv/monitor': 4120989}
    
    class_weights = get_balancing_class_weights(CLASSES, CLASSES_PIXEL_COUNT_DICT)
    print(class_weights)

    train_inputs = {'image': sorted(glob(os.path.join(x_train_dir, '*'))), 'mask': sorted(glob(os.path.join(y_train_dir, '*')))}
    valid_inputs = {'image': sorted(glob(os.path.join(x_valid_dir, '*'))), 'mask': sorted(glob(os.path.join(y_valid_dir, '*')))}

    train_dataset = aw.TFBaseDataset(train_inputs)
    train_dataset = aw.LoadImage(train_dataset, ['image', 'mask'], -1)
    train_dataset = aw.OneHot(train_dataset, ['mask'], list(range(len(CLASSES))))
    train_dataset = aw.Augmentation(train_dataset, get_training_augmentation(HEIGHT, WIDTH, train_dataset))
    # train_dataset = aw.NormalizeImage(train_dataset, ['image'], (0, 1))

    valid_dataset = aw.TFBaseDataset(valid_inputs)
    valid_dataset = aw.LoadImage(valid_dataset, ['image', 'mask'], -1)
    valid_dataset = aw.OneHot(valid_dataset, ['mask'], list(range(len(CLASSES))))
    valid_dataset = aw.Augmentation(valid_dataset, get_validation_augmentation(HEIGHT, WIDTH))
    # valid_dataset = aw.NormalizeImage(valid_dataset, ['image'], (0, 1))

    valid_images = []
    for idx in range(5):
        valid_image = valid_dataset[idx]["image"]
        valid_images.append(valid_image)

    TrainSet = TFDataGenerator(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    ValidationSet = TFDataGenerator(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

    callbacks = [DisplayCallback(),
                #  tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True),
                #  tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=ES_PATIENT, verbose=1),
                 tf.keras.callbacks.ModelCheckpoint(f"{SAVE_PATH}/{SAVE_NAME}/best.ckpt", monitor='val_loss', verbose=1, mode="max", save_best_only=True, save_weights_only=True)]

    model = get_model()
    model.fit(TrainSet,
              epochs=EPOCHS,
              validation_data=ValidationSet,
              callbacks=callbacks)

    plot_predictions(valid_images, COLORMAP, model=model)
    
    run_model = tf.function(lambda x : model(x))
    BATCH_SIZE = 1
    concrete_func = run_model.get_concrete_function(tf.TensorSpec([BATCH_SIZE, HEIGHT, WIDTH, 3], model.inputs[0].dtype))
    tf.saved_model.save(model, f'{SAVE_PATH}/{SAVE_NAME}/saved_model', signatures=concrete_func)