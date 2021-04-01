import pathlib, random, cv2
import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K

input_shape = (224, 224, 3)

def residual_block(x, filters, kernel_size=3, stride=1, conv_shortcut=True, name=None):
    if conv_shortcut:
        shortcut = tf.keras.layers.Conv2D(4 * filters, 1, strides=stride, name=name+'_0_conv')(x)
        shortcut = tf.keras.layers.BatchNormalization(axis=3, epsilon=1.001e-5, name=name+'_0_bn')(shortcut)

    else:
        shortcut = x

    x = tf.keras.layers.Conv2D(filters, 1, strides=stride, name=name + '_1_conv')(x)
    x = tf.keras.layers.BatchNormalization(axis=3, epsilon=1.001e-5, name=name + '_1_bn')(x)
    x = tf.keras.layers.Activation('relu', name=name + '_1_relu')(x)

    x = tf.keras.layers.Conv2D(filters, kernel_size, padding='SAME', name=name + '_2_conv')(x)
    x = tf.keras.layers.BatchNormalization(axis=3, epsilon=1.001e-5, name=name + '_2_bn')(x)
    x = tf.keras.layers.Activation('relu', name=name + '_2_relu')(x)

    x = tf.keras.layers.Conv2D(4 * filters, 1, name=name + '_3_conv')(x)
    x = tf.keras.layers.BatchNormalization(axis=3, epsilon=1.001e-5, name=name + '_3_bn')(x)

    x = tf.keras.layers.Add(name=name + '_add')([shortcut, x])
    x = tf.keras.layers.Activation('relu', name=name + '_out')(x)

    return x

def residual_stack(x, filters, blocks, stride1=2, name=None):
    x = residual_block(x, filters, stride=stride1, name=name + '_block1')

    for i in range(2, blocks + 1):
        x = residual_block(x, filters, conv_shortcut=False, name=name + '_block' + str(i))

    return x

inputs = tf.keras.layers.Input(shape=input_shape)
x = tf.keras.layers.ZeroPadding2D(padding=((3, 3), (3, 3)), name='conv1_pad')(inputs)
x = tf.keras.layers.Conv2D(64, 7, strides=2, use_bias=True, name='conv1_conv')(x)
x = tf.keras.layers.BatchNormalization(axis=3, epsilon=1.001e-5, name='conv1_bn')(x)
x = tf.keras.layers.Activation('relu', name='conv1_relu')(x)

x = tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad')(x)
x = tf.keras.layers.MaxPooling2D(3, strides=2, name='pool1_pool')(x)

x = residual_stack(x, 64, 3, stride1=1, name='conv2')
x = residual_stack(x, 128, 4, name='conv3')
x = residual_stack(x, 256, 6, name='conv4')
x = residual_stack(x, 512, 3, name='conv5')

x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
x = tf.keras.layers.Dense(1000, activation='softmax', name='predictions')(x)

model = tf.keras.Model(inputs=inputs, outputs=x)
model.summary()
tf.keras.utils.plot_model(model, show_shapes=True)