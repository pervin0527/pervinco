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

from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import DepthwiseConv2D
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.utils import get_source_inputs
from tensorflow.keras.utils import get_file
from tensorflow.keras import backend as K
from tensorflow.keras.applications.imagenet_utils import preprocess_input


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

WEIGHTS_PATH_X = "https://github.com/bonlime/keras-deeplab-v3-plus/releases/download/1.1/deeplabv3_xception_tf_dim_ordering_tf_kernels.h5"
WEIGHTS_PATH_MOBILE = "https://github.com/bonlime/keras-deeplab-v3-plus/releases/download/1.1/deeplabv3_mobilenetv2_tf_dim_ordering_tf_kernels.h5"
WEIGHTS_PATH_X_CS = "https://github.com/bonlime/keras-deeplab-v3-plus/releases/download/1.2/deeplabv3_xception_tf_dim_ordering_tf_kernels_cityscapes.h5"
WEIGHTS_PATH_MOBILE_CS = "https://github.com/bonlime/keras-deeplab-v3-plus/releases/download/1.2/deeplabv3_mobilenetv2_tf_dim_ordering_tf_kernels_cityscapes.h5"


def SepConv_BN(x, filters, prefix, stride=1, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3):
    if stride == 1:
        depth_padding = 'same'
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding2D((pad_beg, pad_end))(x)
        depth_padding = 'valid'

    if not depth_activation:
        x = Activation(tf.nn.relu)(x)

    x = DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride), dilation_rate=(rate, rate), padding=depth_padding, use_bias=False, name=prefix + '_depthwise')(x)
    x = BatchNormalization(name=prefix + '_depthwise_BN', epsilon=epsilon)(x)
    
    if depth_activation:
        x = Activation(tf.nn.relu)(x)
    x = Conv2D(filters, (1, 1), padding='same', use_bias=False, name=prefix + '_pointwise')(x)
    x = BatchNormalization(name=prefix + '_pointwise_BN', epsilon=epsilon)(x)

    if depth_activation:
        x = Activation(tf.nn.relu)(x)

    return x


def _conv2d_same(x, filters, prefix, stride=1, kernel_size=3, rate=1):
    if stride == 1:
        return Conv2D(filters, (kernel_size, kernel_size), strides=(stride, stride), padding='same', use_bias=False, dilation_rate=(rate, rate), name=prefix)(x)
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding2D((pad_beg, pad_end))(x)
        return Conv2D(filters, (kernel_size, kernel_size), strides=(stride, stride), padding='valid', use_bias=False, dilation_rate=(rate, rate), name=prefix)(x)


def _xception_block(inputs, depth_list, prefix, skip_connection_type, stride, rate=1, depth_activation=False, return_skip=False):
    residual = inputs
    for i in range(3):
        residual = SepConv_BN(residual, depth_list[i], prefix + '_separable_conv{}'.format(i + 1), stride=stride if i == 2 else 1, rate=rate, depth_activation=depth_activation)
        if i == 1:
            skip = residual
    if skip_connection_type == 'conv':
        shortcut = _conv2d_same(inputs, depth_list[-1], prefix + '_shortcut', kernel_size=1, stride=stride)
        shortcut = BatchNormalization(name=prefix + '_shortcut_BN')(shortcut)
        outputs = layers.add([residual, shortcut])

    elif skip_connection_type == 'sum':
        outputs = layers.add([residual, inputs])
    
    elif skip_connection_type == 'none':
        outputs = residual
    
    if return_skip:
        return outputs, skip
    
    else:
        return outputs


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _inverted_res_block(inputs, expansion, stride, alpha, filters, block_id, skip_connection, rate=1):
    in_channels = inputs.shape[-1].value  # inputs._keras_shape[-1]
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
    x = inputs
    prefix = 'expanded_conv_{}_'.format(block_id)
    if block_id:
        # Expand

        x = Conv2D(expansion * in_channels, kernel_size=1, padding='same', use_bias=False, activation=None, name=prefix + 'expand')(x)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999, name=prefix + 'expand_BN')(x)
        x = Activation(tf.nn.relu6, name=prefix + 'expand_relu')(x)
    else:
        prefix = 'expanded_conv_'
    # Depthwise
    x = DepthwiseConv2D(kernel_size=3, strides=stride, activation=None, use_bias=False, padding='same', dilation_rate=(rate, rate), name=prefix + 'depthwise')(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999, name=prefix + 'depthwise_BN')(x)

    x = Activation(tf.nn.relu6, name=prefix + 'depthwise_relu')(x)

    # Project
    x = Conv2D(pointwise_filters, kernel_size=1, padding='same', use_bias=False, activation=None, name=prefix + 'project')(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999, name=prefix + 'project_BN')(x)

    if skip_connection:
        return Add(name=prefix + 'add')([inputs, x])

    # if in_channels == pointwise_filters and stride == 1:
    #    return Add(name='res_connect_' + str(block_id))([inputs, x])

    return x


def Deeplabv3(weights='pascal_voc', input_tensor=None, input_shape=(512, 512, 3), classes=21, backbone='mobilenetv2', OS=16, alpha=1., activation=None):
    if not (weights in {'pascal_voc', 'cityscapes', None}):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `pascal_voc`, or `cityscapes` '
                         '(pre-trained on PASCAL VOC)')

    if not (backbone in {'xception', 'mobilenetv2'}):
        raise ValueError('The `backbone` argument should be either '
                         '`xception`  or `mobilenetv2` ')

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        img_input = input_tensor

    if backbone == 'xception':
        if OS == 8:
            entry_block3_stride = 1
            middle_block_rate = 2  # ! Not mentioned in paper, but required
            exit_block_rates = (2, 4)
            atrous_rates = (12, 24, 36)
        else:
            entry_block3_stride = 2
            middle_block_rate = 1
            exit_block_rates = (1, 2)
            atrous_rates = (6, 12, 18)

        x = Conv2D(32, (3, 3), strides=(2, 2), name='entry_flow_conv1_1', use_bias=False, padding='same')(img_input)
        x = BatchNormalization(name='entry_flow_conv1_1_BN')(x)
        x = Activation(tf.nn.relu)(x)

        x = _conv2d_same(x, 64, 'entry_flow_conv1_2', kernel_size=3, stride=1)
        x = BatchNormalization(name='entry_flow_conv1_2_BN')(x)
        x = Activation(tf.nn.relu)(x)

        x = _xception_block(x, [128, 128, 128], 'entry_flow_block1', skip_connection_type='conv', stride=2, depth_activation=False)
        x, skip1 = _xception_block(x, [256, 256, 256], 'entry_flow_block2', skip_connection_type='conv', stride=2, depth_activation=False, return_skip=True)

        x = _xception_block(x, [728, 728, 728], 'entry_flow_block3', skip_connection_type='conv', stride=entry_block3_stride, depth_activation=False)
        for i in range(16):
            x = _xception_block(x, [728, 728, 728], 'middle_flow_unit_{}'.format(i + 1), skip_connection_type='sum', stride=1, rate=middle_block_rate, depth_activation=False)

        x = _xception_block(x, [728, 1024, 1024], 'exit_flow_block1', skip_connection_type='conv', stride=1, rate=exit_block_rates[0], depth_activation=False)
        x = _xception_block(x, [1536, 1536, 2048], 'exit_flow_block2', skip_connection_type='none', stride=1, rate=exit_block_rates[1], depth_activation=True)

    else:
        OS = 8
        first_block_filters = _make_divisible(32 * alpha, 8)
        x = Conv2D(first_block_filters, kernel_size=3, strides=(2, 2), padding='same', use_bias=False, name='Conv' if input_shape[2] == 3 else 'Conv_')(img_input)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999, name='Conv_BN')(x)
        x = Activation(tf.nn.relu6, name='Conv_Relu6')(x)

        x = _inverted_res_block(x, filters=16, alpha=alpha, stride=1, expansion=1, block_id=0, skip_connection=False)

        x = _inverted_res_block(x, filters=24, alpha=alpha, stride=2, expansion=6, block_id=1, skip_connection=False)
        x = _inverted_res_block(x, filters=24, alpha=alpha, stride=1, expansion=6, block_id=2, skip_connection=True)

        x = _inverted_res_block(x, filters=32, alpha=alpha, stride=2, expansion=6, block_id=3, skip_connection=False)
        x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1, expansion=6, block_id=4, skip_connection=True)
        x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1, expansion=6, block_id=5, skip_connection=True)

        # stride in block 6 changed from 2 -> 1, so we need to use rate = 2
        x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, expansion=6, block_id=6, skip_connection=False)
        x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, rate=2, expansion=6, block_id=7, skip_connection=True)
        x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, rate=2, expansion=6, block_id=8, skip_connection=True)
        x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, rate=2, expansion=6, block_id=9, skip_connection=True)

        x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=2, expansion=6, block_id=10, skip_connection=False)
        x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=2, expansion=6, block_id=11, skip_connection=True)
        x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=2, expansion=6, block_id=12, skip_connection=True)

        x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, rate=2, expansion=6, block_id=13, skip_connection=False)
        x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, rate=4, expansion=6, block_id=14, skip_connection=True)
        x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, rate=4, expansion=6, block_id=15, skip_connection=True)

        x = _inverted_res_block(x, filters=320, alpha=alpha, stride=1, rate=4, expansion=6, block_id=16, skip_connection=False)

    shape_before = tf.shape(x)
    b4 = GlobalAveragePooling2D()(x)
    b4_shape = tf.keras.backend.int_shape(b4)
    b4 = Reshape((1, 1, b4_shape[1]))(b4)
    b4 = Conv2D(256, (1, 1), padding='same', use_bias=False, name='image_pooling')(b4)
    b4 = BatchNormalization(name='image_pooling_BN', epsilon=1e-5)(b4)
    b4 = Activation(tf.nn.relu)(b4)
    size_before = tf.keras.backend.int_shape(x)
    b4 = tf.keras.layers.experimental.preprocessing.Resizing(*size_before[1:3], interpolation="bilinear")(b4)
    
    b0 = Conv2D(256, (1, 1), padding='same', use_bias=False, name='aspp0')(x)
    b0 = BatchNormalization(name='aspp0_BN', epsilon=1e-5)(b0)
    b0 = Activation(tf.nn.relu, name='aspp0_activation')(b0)

    if backbone == 'xception':
        b1 = SepConv_BN(x, 256, 'aspp1', rate=atrous_rates[0], depth_activation=True, epsilon=1e-5)
        b2 = SepConv_BN(x, 256, 'aspp2', rate=atrous_rates[1], depth_activation=True, epsilon=1e-5)
        b3 = SepConv_BN(x, 256, 'aspp3', rate=atrous_rates[2], depth_activation=True, epsilon=1e-5)

        x = Concatenate()([b4, b0, b1, b2, b3])
    else:
        x = Concatenate()([b4, b0])

    x = Conv2D(256, (1, 1), padding='same', use_bias=False, name='concat_projection')(x)
    x = BatchNormalization(name='concat_projection_BN', epsilon=1e-5)(x)
    x = Activation(tf.nn.relu)(x)
    x = Dropout(0.1)(x)

    if backbone == 'xception':
        skip_size = tf.keras.backend.int_shape(skip1)
        x = tf.keras.layers.experimental.preprocessing.Resizing(*skip_size[1:3], interpolation="bilinear")(x)
        dec_skip1 = Conv2D(48, (1, 1), padding='same', use_bias=False, name='feature_projection0')(skip1)
        dec_skip1 = BatchNormalization(name='feature_projection0_BN', epsilon=1e-5)(dec_skip1)
        dec_skip1 = Activation(tf.nn.relu)(dec_skip1)
        x = Concatenate()([x, dec_skip1])
        x = SepConv_BN(x, 256, 'decoder_conv0', depth_activation=True, epsilon=1e-5)
        x = SepConv_BN(x, 256, 'decoder_conv1', depth_activation=True, epsilon=1e-5)

    if (weights == 'pascal_voc' and classes == 21) or (weights == 'cityscapes' and classes == 19):
        last_layer_name = 'logits_semantic'
    else:
        last_layer_name = 'custom_logits_semantic'

    x = Conv2D(classes, (1, 1), padding='same', name=last_layer_name)(x)
    size_before3 = tf.keras.backend.int_shape(img_input)
    x = tf.keras.layers.experimental.preprocessing.Resizing(*size_before3[1:3], interpolation="bilinear")(x)

    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    if activation in {'softmax', 'sigmoid'}:
        x = tf.keras.layers.Activation(activation)(x)

    model = Model(inputs, x, name='deeplabv3plus')

    # if weights == 'pascal_voc':
    #     if backbone == 'xception':
    #         weights_path = get_file('deeplabv3_xception_tf_dim_ordering_tf_kernels.h5',
    #                                 WEIGHTS_PATH_X,
    #                                 cache_subdir='models')
    #     else:
    #         weights_path = get_file('deeplabv3_mobilenetv2_tf_dim_ordering_tf_kernels.h5',
    #                                 WEIGHTS_PATH_MOBILE,
    #                                 cache_subdir='models')
    #     model.load_weights(weights_path, by_name=True)
    # elif weights == 'cityscapes':
    #     if backbone == 'xception':
    #         weights_path = get_file('deeplabv3_xception_tf_dim_ordering_tf_kernels_cityscapes.h5',
    #                                 WEIGHTS_PATH_X_CS,
    #                                 cache_subdir='models')
    #     else:
    #         weights_path = get_file('deeplabv3_mobilenetv2_tf_dim_ordering_tf_kernels_cityscapes.h5',
    #                                 WEIGHTS_PATH_MOBILE_CS,
    #                                 cache_subdir='models')
    #     model.load_weights(weights_path, by_name=True)

    return model

def preprocess_input(x):
    """Preprocesses a numpy array encoding a batch of images.
    # Arguments
        x: a 4D numpy array consists of RGB values within [0, 255].
    # Returns
        Input array scaled to [-1.,1.]
    """
    return preprocess_input(x, mode='tf')


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

def get_model():
    with strategy.scope(): 
        dice_loss = advisor.losses.DiceLoss(class_weights=None)
        categorical_focal_loss = advisor.losses.CategoricalFocalLoss()
        loss = dice_loss + (1 * categorical_focal_loss)
        metrics = [advisor.metrics.FScore(threshold=0.5), advisor.metrics.IOUScore(threshold=0.5)]
        optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
        
        # model = DeeplabV3Plus(len(CLASSES))
        model = Deeplabv3(weights=None, input_tensor=None, input_shape=(HEIGHT, WIDTH, 3), classes=len(CLASSES), backbone='xception', OS=16, alpha=1., activation="softmax")
        model.summary()
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
    EPOCHS = 100
    BATCH_SIZE = 8
    ES_PATIENT = 10
    HEIGHT, WIDTH = 512, 512
    BACKBONE_NAME = "Xception"
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
                 tf.keras.callbacks.ModelCheckpoint(f"{SAVE_PATH}/{SAVE_NAME}/best.ckpt", monitor='val_iou_score', verbose=1, mode="max", save_best_only=True, save_weights_only=True)]

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
