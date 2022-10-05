import tensorflow as tf
from functools import wraps
from loss import get_yolo_loss
from data_utils import compose
from tensorflow.keras import backend as K


class Focus(tf.keras.layers.Layer):
    def __init__(self):
        super(Focus, self).__init__()

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] // 2 if input_shape[1] != None else input_shape[1], input_shape[2] // 2 if input_shape[2] != None else input_shape[2], input_shape[3] * 4)

    def call(self, x):
        return tf.concat([x[...,  ::2,  ::2, :],
                          x[..., 1::2,  ::2, :],
                          x[...,  ::2, 1::2, :],
                          x[..., 1::2, 1::2, :]], axis=-1)


class SiLU(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(SiLU, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        return inputs * K.sigmoid(inputs)

    def get_config(self):
        config = super(SiLU, self).get_config()
        return config

    def compute_output_shape(self, input_shape):
        return input_shape


@wraps(tf.keras.layers.Conv2D)
def Darknet_Conv2D(*args, **kwargs):
    darknet_conv_kwargs = {'kernel_initializer' : tf.keras.initializers.RandomNormal(stddev=0.02), 'kernel_regularizer' : tf.keras.regularizers.l2(kwargs.get('weight_decay', 5e-4))}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2, 2) else 'same'
    try:
        del kwargs['weight_decay']
    except:
        pass
    darknet_conv_kwargs.update(kwargs)

    return tf.keras.layers.Conv2D(*args, **darknet_conv_kwargs)


def Darknet_Conv2D_BN_SiLU(*args, **kwargs):
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    if "name" in kwargs.keys():
        no_bias_kwargs['name'] = kwargs['name'] + '.conv'
    return compose(
        Darknet_Conv2D(*args, **no_bias_kwargs),
        tf.keras.layers.BatchNormalization(momentum = 0.97, epsilon = 0.001, name = kwargs['name'] + '.bn'),
        SiLU())


def Darknet_Conv2D_BN_SiLU(*args, **kwargs):
    no_bias_kwargs = {"use_bias" : False}
    no_bias_kwargs.update(kwargs)

    if "name" in kwargs.keys():
        no_bias_kwargs["name"] = kwargs["name"] + ".conv"
    
    return compose(Darknet_Conv2D(*args, **no_bias_kwargs), 
                   tf.keras.layers.BatchNormalization(momentum=0.97, epsilon=0.001, name=kwargs["name"] + ".bn"),
                   SiLU())


def SPPBottleneck(x, out_channels, weight_decay=5e-4, name = ""):
    x = Darknet_Conv2D_BN_SiLU(out_channels // 2, (1, 1), weight_decay=weight_decay, name = name + '.conv1')(x)
    maxpool1 = tf.keras.layers.MaxPooling2D(pool_size=(5, 5), strides=(1, 1), padding='same')(x)
    maxpool2 = tf.keras.layers.MaxPooling2D(pool_size=(9, 9), strides=(1, 1), padding='same')(x)
    maxpool3 = tf.keras.layers.MaxPooling2D(pool_size=(13, 13), strides=(1, 1), padding='same')(x)
    x = tf.keras.layers.Concatenate()([x, maxpool1, maxpool2, maxpool3])
    x = Darknet_Conv2D_BN_SiLU(out_channels, (1, 1), weight_decay=weight_decay, name = name + '.conv2')(x)
    return x


def Bottleneck(x, out_channels, shortcut=True, weight_decay=5e-4, name = ""):
    y = compose(
            Darknet_Conv2D_BN_SiLU(out_channels, (1, 1), weight_decay=weight_decay, name = name + '.conv1'),
            Darknet_Conv2D_BN_SiLU(out_channels, (3, 3), weight_decay=weight_decay, name = name + '.conv2'))(x)
    if shortcut:
        y = tf.keras.layers.Add()([x, y])
    return y


def CSPLayer(x, num_filters, num_blocks, shortcut=True, expansion=0.5, weight_decay=5e-4, name=""):
    hidden_channels = int(num_filters * expansion)

    x_1 = Darknet_Conv2D_BN_SiLU(hidden_channels, (1, 1), weight_decay=weight_decay, name = name + '.conv1')(x)
    x_2 = Darknet_Conv2D_BN_SiLU(hidden_channels, (1, 1), weight_decay=weight_decay, name = name + '.conv2')(x)

    for i in range(num_blocks):
        x_1 = Bottleneck(x_1, hidden_channels, shortcut=shortcut, weight_decay=weight_decay, name = name + '.m.' + str(i))

    route = tf.keras.layers.Concatenate()([x_1, x_2])

    return Darknet_Conv2D_BN_SiLU(num_filters, (1, 1), weight_decay=weight_decay, name = name + '.conv3')(route)


def residual_block(x, num_filters, num_blocks, expansion=0.5, shortcut=True, last=False, weight_decay=5e-4, name = ""):
    x = tf.keras.layers.ZeroPadding2D(((1, 0),(1, 0)))(x)
    x = Darknet_Conv2D_BN_SiLU(num_filters, (3, 3), strides = (2, 2), weight_decay=weight_decay, name = name + '.0')(x)
    if last:
        x = SPPBottleneck(x, num_filters, weight_decay=weight_decay, name = name + '.1')
    return CSPLayer(x, num_filters, num_blocks, shortcut=shortcut, expansion=expansion, weight_decay=weight_decay, name = name + '.1' if not last else name + '.2')


def darknet(x, depth_mul, width_mul, weight_decay=5e-4):
    base_channels = int(width_mul * 64)
    base_depth = max(round(depth_mul * 3), 1)

    x = Focus()(x)
    x = Darknet_Conv2D_BN_SiLU(base_channels, (3, 3), weight_decay=weight_decay, name="backbone.bacnbone.stem.conv")(x)
    x = residual_block(x, base_channels * 2, base_depth, weight_decay=weight_decay, name = 'backbone.backbone.dark2')
    x = residual_block(x, base_channels * 4, base_depth * 3, weight_decay=weight_decay, name = 'backbone.backbone.dark3')
    feature1 = x

    x = residual_block(x, base_channels * 8, base_depth * 3, weight_decay=weight_decay, name = 'backbone.backbone.dark4')
    feature2 = x

    x = residual_block(x, base_channels * 16, base_depth, shortcut=False, last=True, weight_decay=weight_decay, name = 'backbone.backbone.dark5')
    feature3 = x

    return feature1, feature2, feature3


def yolo_base(input_shape, num_classes, phi, weight_decay=5e-4):
    depth_dict = {"tiny":0.33, "s":0.33, "m":0.67, "l":1.00, "x":1.33,}
    width_dict = {"tiny":0.375, "s":0.50, "m":0.75, "l":1.00, "x":1.25,}
    depth, width = depth_dict[phi], width_dict[phi]
    in_channels = [256, 512, 1024]

    inputs = tf.keras.Input(input_shape)
    feature1, feature2, feature3 = darknet(inputs, depth, width, weight_decay=weight_decay)

    P5 = Darknet_Conv2D_BN_SiLU(int(in_channels[1] * width), (1, 1), weight_decay=weight_decay, name = 'backbone.lateral_conv0')(feature3)
    P5_upsample = tf.keras.layers.UpSampling2D()(P5)
    P5_upsample = tf.keras.layers.Concatenate(axis = -1)([P5_upsample, feature2])
    P5_upsample = CSPLayer(P5_upsample, int(in_channels[1] * width), round(3 * depth), shortcut = False, weight_decay=weight_decay, name = 'backbone.C3_p4')

    P4 = Darknet_Conv2D_BN_SiLU(int(in_channels[0] * width), (1, 1), weight_decay=weight_decay, name = 'backbone.reduce_conv1')(P5_upsample)
    P4_upsample = tf.keras.layers.UpSampling2D()(P4)
    P4_upsample = tf.keras.layers.Concatenate(axis = -1)([P4_upsample, feature1])
    P3_out = CSPLayer(P4_upsample, int(in_channels[0] * width), round(3 * depth), shortcut = False, weight_decay=weight_decay, name = 'backbone.C3_p3')

    P3_downsample = tf.keras.layers.ZeroPadding2D(((1, 0),(1, 0)))(P3_out)
    P3_downsample = Darknet_Conv2D_BN_SiLU(int(in_channels[0] * width), (3, 3), strides = (2, 2), weight_decay=weight_decay, name = 'backbone.bu_conv2')(P3_downsample)
    P3_downsample = tf.keras.layers.Concatenate(axis = -1)([P3_downsample, P4])
    P4_out = CSPLayer(P3_downsample, int(in_channels[1] * width), round(3 * depth), shortcut = False, weight_decay=weight_decay, name = 'backbone.C3_n3')

    P4_downsample = tf.keras.layers.ZeroPadding2D(((1, 0),(1, 0)))(P4_out)
    P4_downsample = Darknet_Conv2D_BN_SiLU(int(in_channels[1] * width), (3, 3), strides = (2, 2), weight_decay=weight_decay, name = 'backbone.bu_conv1')(P4_downsample)
    P4_downsample = tf.keras.layers.Concatenate(axis = -1)([P4_downsample, P5])
    P5_out = CSPLayer(P4_downsample, int(in_channels[2] * width), round(3 * depth), shortcut = False, weight_decay=weight_decay, name = 'backbone.C3_n4')

    fpn_outs = [P3_out, P4_out, P5_out]

    yolo_outs = []
    for i, out in enumerate(fpn_outs):
        stem = Darknet_Conv2D_BN_SiLU(int(256 * width), (1, 1), strides = (1, 1), weight_decay=weight_decay, name = 'head.stems.' + str(i))(out)
        cls_conv = Darknet_Conv2D_BN_SiLU(int(256 * width), (3, 3), strides = (1, 1), weight_decay=weight_decay, name = 'head.cls_convs.' + str(i) + '.0')(stem)
        cls_conv = Darknet_Conv2D_BN_SiLU(int(256 * width), (3, 3), strides = (1, 1), weight_decay=weight_decay, name = 'head.cls_convs.' + str(i) + '.1')(cls_conv)
        cls_pred = Darknet_Conv2D(num_classes, (1, 1), strides = (1, 1), weight_decay=weight_decay, name = 'head.cls_preds.' + str(i))(cls_conv)
        reg_conv = Darknet_Conv2D_BN_SiLU(int(256 * width), (3, 3), strides = (1, 1), weight_decay=weight_decay, name = 'head.reg_convs.' + str(i) + '.0')(stem)
        reg_conv = Darknet_Conv2D_BN_SiLU(int(256 * width), (3, 3), strides = (1, 1), weight_decay=weight_decay, name = 'head.reg_convs.' + str(i) + '.1')(reg_conv)
        reg_pred = Darknet_Conv2D(4, (1, 1), strides = (1, 1), weight_decay=weight_decay, name = 'head.reg_preds.' + str(i))(reg_conv)
        obj_pred = Darknet_Conv2D(1, (1, 1), strides = (1, 1), weight_decay=weight_decay, name = 'head.obj_preds.' + str(i))(reg_conv)
        output   = tf.keras.layers.Concatenate(axis = -1)([reg_pred, obj_pred, cls_pred])
        yolo_outs.append(output)
        
    return tf.keras.Model(inputs, yolo_outs)


def get_train_model(model_body, input_shape, num_classes):
    y_true = [tf.keras.Input(shape = (None, 5))]
    model_loss  = tf.keras.layers.Lambda(get_yolo_loss(input_shape, len(model_body.output), num_classes), 
                                         output_shape = (1, ), 
                                         name = 'yolo_loss',)([*model_body.output, *y_true])
    
    model = tf.keras.Model([model_body.input, *y_true], model_loss)

    return model