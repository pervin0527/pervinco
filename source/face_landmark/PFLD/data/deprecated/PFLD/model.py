import tensorflow as tf
from losses import PFLDLoss, valid_loss
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import  Conv2D, BatchNormalization, Activation, DepthwiseConv2D, Add, AvgPool2D, Concatenate, Dense, Input, Reshape, MaxPool2D, Flatten

def relu6(x):
    return K.relu(x, max_value=6)


def conv_bn(filters, kernel_size, strides, padding='same'):
    def _conv_bn(x):
        x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=False)(x)
        x = BatchNormalization()(x)
        x = relu6(x)
        return x
    
    return _conv_bn


def InvertedResidual(filters, strides, use_res_connect, expand_ratio=6,name=''):
    def _InvertedResidual(inputs):
        x = Conv2D(filters=filters*expand_ratio, kernel_size=1, strides=1, padding='valid', use_bias=False)(inputs)
        x = BatchNormalization()(x)
        x = relu6(x)
        x = DepthwiseConv2D(kernel_size=3, strides=strides, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = relu6(x)
        x = Conv2D(filters=filters, kernel_size=1, strides=1, padding='valid', use_bias=False)(x)
        x = BatchNormalization()(x)

        if use_res_connect:
            if name:
                x = Add(name=name)([inputs, x])
                return x
            else:
                x = Add()([inputs, x])
                return x
        else:
            return x
    
    return _InvertedResidual


def PFLDInference(inputs, is_train=True, keypoints=196):
    inputs = Input(shape=inputs)
    x = Conv2D(filters=64, kernel_size=3, strides=2, padding='same', use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = relu6(x)

    x = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = relu6(x)

    x = InvertedResidual(64, 2, False, 2)(x)

    x = InvertedResidual(64, 1, True, 2)(x)
    x = InvertedResidual(64, 1, True, 2)(x)
    x = InvertedResidual(64, 1, True, 2)(x)

    out1 = InvertedResidual(64, 1, True, 2)(x)
 
    x = InvertedResidual(128, 2, False, 2)(out1)
    
    x = InvertedResidual(128, 1, False, 4)(x)
    x = InvertedResidual(128, 1, True, 4)(x)
    x = InvertedResidual(128, 1, True, 4)(x)
    x = InvertedResidual(128, 1, True, 4)(x)
    x = InvertedResidual(128, 1, True, 4)(x)
    x = InvertedResidual(128, 1, True, 4)(x)
    
    x = InvertedResidual(16, 1, False, 2)(x)

    x1 = AvgPool2D(pool_size=(14, 14))(x)
    x1 = Reshape((x1.shape[1]*x1.shape[2]*x1.shape[3],))(x1)
    
    x = conv_bn(32, 3, 2, padding='same')(x)
    x2 = AvgPool2D(pool_size=(7, 7))(x)
    x2 = Reshape((x2.shape[1]*x2.shape[2]*x2.shape[3],))(x2)
    
    x3 = Conv2D(filters=128, kernel_size=7, strides=1, padding='valid')(x)
    x3 = Activation('relu')(x3)
    x3 = Reshape((x3.shape[1]*x3.shape[2]*x3.shape[3],))(x3)
    
    multi_scale = Concatenate()([x1, x2, x3])
    landmarks = Dense(keypoints, name='landmarks')(multi_scale)
    
    if is_train:
        out1 = AuxiliaryNet(out1)
        return Model(inputs, [Concatenate(name='train_out')([landmarks, out1]), landmarks])
    else:
        return Model(inputs, landmarks)


def AuxiliaryNet(inputs):
    x = conv_bn(128, 3, 2)(inputs)
    x = conv_bn(128, 3, 1)(x)
    x = conv_bn(32, 3, 2)(x)
    x = conv_bn(128, 7, 1)(x)
    x = MaxPool2D(pool_size=(3, 3))(x)
    x = Flatten()(x)
    x = Dense(32)(x)
    x = Dense(3, name='out1')(x)
    return x