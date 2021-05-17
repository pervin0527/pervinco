import os, sys
import numpy as np
import tensorflow as tf

def dense_bn(x, units, use_bias=True, scope=None, activation=None):
    with tf.keras.backend.name_scope(scope):
        x = tf.keras.layers.Dense(units=units, use_bias=use_bias)(x)
        x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
        x = tf.keras.layers.Activation(activation)(x)
    return x

def conv1d_bn(x, num_filters, kernel_size, padding='same', strides=1, use_bias=False, scope=None, activation='relu'):
    with tf.keras.backend.name_scope(scope):
        input_shape = x.get_shape().as_list()[-2:]
        x = tf.keras.layers.Conv1D(num_filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias, input_shape=input_shape)(x)
        x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
        x = tf.keras.layers.Activation(activation)(x)
    return x

def transform_net(inputs, scope=None, regularize=False):
    with tf.keras.backend.name_scope(scope):

        input_shape = inputs.get_shape().as_list()
        k = input_shape[-1]

        net = conv1d_bn(inputs, num_filters=64, kernel_size=1, padding='valid', use_bias=True, scope='tconv1')
        net = conv1d_bn(net, num_filters=128, kernel_size=1, padding='valid', use_bias=True, scope='tconv2')
        net = conv1d_bn(net, num_filters=1024, kernel_size=1, padding='valid', use_bias=True, scope='tconv3')

        net = tf.keras.layers.GlobalMaxPooling1D(data_format='channels_last')(net)

        net = dense_bn(net, units=512, scope='tfc1', activation='relu')
        net = dense_bn(net, units=256, scope='tfc2', activation='relu')

        transform = tf.keras.layers.Dense(units=k * k, kernel_initializer='zeros', bias_initializer=tf.keras.initializers.Constant(np.eye(k).flatten()), activity_regularizer=None)(net) # OrthogonalRegularizer(l2=0.001) if regularize else
        transform = tf.keras.layers.Reshape((k, k))(transform)

    return transform

def get_base_model(inputs):
    ptransform = transform_net(inputs, scope='transform_net1', regularize=False)
    point_cloud_transformed = tf.keras.layers.Dot(axes=(2, 1))([inputs, ptransform])

    net = conv1d_bn(point_cloud_transformed, num_filters=64, kernel_size=1, padding='valid', use_bias=True, scope='conv1')
    net = conv1d_bn(net, num_filters=64, kernel_size=1, padding='valid', use_bias=True, scope='conv2')

    ftransform = transform_net(net, scope='transform_net2', regularize=True)
    net_transformed = tf.keras.layers.Dot(axes=(2, 1))([net, ftransform])

    # Second block of convolutions
    net = conv1d_bn(net_transformed, num_filters=64, kernel_size=1, padding='valid', use_bias=True, scope='conv3')
    net = conv1d_bn(net, num_filters=128, kernel_size=1, padding='valid', use_bias=True, scope='conv4')
    hx = conv1d_bn(net, num_filters=1024, kernel_size=1, padding='valid', use_bias=True, scope='hx')

    # add Maxpooling here, because it is needed in both nets.
    net = tf.keras.layers.GlobalMaxPooling1D(data_format='channels_last', name='maxpool')(hx)

    return net, net_transformed

def get_model(input_shape, classes, activation=None):
    assert tf.keras.backend.image_data_format() == 'channels_last'
    inputs = tf.keras.Input(input_shape, name='Input_cloud')
    maxpool, _ = get_base_model(inputs)

    # Top layers
    if isinstance(classes, dict):
        # Fully connected layers
        net = [dense_bn(maxpool, units=512, scope=r + '_fc1', activation='relu') for r in classes]
        net = [tf.keras.layers.Dropout(0.3, name=r + '_dp1')(n) for r, n in zip(classes, net)]
        net = [dense_bn(n, units=256, scope=r + '_fc2', activation='relu') for r, n in zip(classes, net)]
        net = [tf.keras.layers.Dropout(0.3, name=r + '_dp2')(n) for r, n in zip(classes, net)]
        net = [tf.keras.layers.Dense(units=classes[r], activation=activation, name=r)(n) for r, n in zip(classes, net)]
    else:
        net = dense_bn(maxpool, units=512, scope='fc1', activation='relu')
        net = tf.keras.layers.Dropout(0.3, name='dp1')(net)
        net = dense_bn(net, units=256, scope='fc2', activation='relu')
        net = tf.keras.layers.Dropout(0.3, name='dp2')(net)
        net = tf.keras.layers.Dense(units=classes, name='fc3', activation=activation)(net)

    model = tf.keras.Model(inputs, net, name='pointnet_cls')

    return model

model = get_model((None, 3), 10)
model.summary()