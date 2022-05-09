import tensorflow as tf

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
        x = tf.keras.layers.ZeroPadding2D((pad_beg, pad_end))(x)
        depth_padding = 'valid'

    if not depth_activation:
        x = tf.keras.layers.Activation(tf.nn.relu)(x)

    x = tf.keras.layers.DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride), dilation_rate=(rate, rate), padding=depth_padding, use_bias=False, name=prefix + '_depthwise')(x)
    x = tf.keras.layers.BatchNormalization(name=prefix + '_depthwise_BN', epsilon=epsilon)(x)
    
    if depth_activation:
        x = tf.keras.layers.Activation(tf.nn.relu)(x)
    x = tf.keras.layers.Conv2D(filters, (1, 1), padding='same', use_bias=False, name=prefix + '_pointwise')(x)
    x = tf.keras.layers.BatchNormalization(name=prefix + '_pointwise_BN', epsilon=epsilon)(x)
    
    if depth_activation:
        x = tf.keras.layers.Activation(tf.nn.relu)(x)

    return x

def _conv2d_same(x, filters, prefix, stride=1, kernel_size=3, rate=1):
    if stride == 1:
        return tf.keras.layers.Conv2D(filters, (kernel_size, kernel_size), strides=(stride, stride), padding='same', use_bias=False, dilation_rate=(rate, rate), name=prefix)(x)

    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = tf.keras.layers.ZeroPadding2D((pad_beg, pad_end))(x)
        
        return tf.keras.layers.Conv2D(filters, (kernel_size, kernel_size), strides=(stride, stride), padding='valid', use_bias=False, dilation_rate=(rate, rate), name=prefix)(x)


def _xception_block(inputs, depth_list, prefix, skip_connection_type, stride, rate=1, depth_activation=False, return_skip=False):
    residual = inputs
    for i in range(3):
        residual = SepConv_BN(residual, depth_list[i], prefix + '_separable_conv{}'.format(i + 1), stride=stride if i == 2 else 1, rate=rate, depth_activation=depth_activation)
        
        if i == 1:
            skip = residual
    if skip_connection_type == 'conv':
        shortcut = _conv2d_same(inputs, depth_list[-1], prefix + '_shortcut', kernel_size=1, stride=stride)
        shortcut = tf.keras.layers.BatchNormalization(name=prefix + '_shortcut_BN')(shortcut)
        outputs = tf.keras.layers.add([residual, shortcut])

    elif skip_connection_type == 'sum':
        outputs = tf.keras.layers.add([residual, inputs])

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

        x = tf.keras.layers.Conv2D(expansion * in_channels, kernel_size=1, padding='same', use_bias=False, activation=None, name=prefix + 'expand')(x)
        x = tf.keras.layers.BatchNormalization(epsilon=1e-3, momentum=0.999, name=prefix + 'expand_BN')(x)
        x = tf.keras.layers.Activation(tf.nn.relu6, name=prefix + 'expand_relu')(x)

    else:
        prefix = 'expanded_conv_'
    # Depthwise
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=stride, activation=None, use_bias=False, padding='same', dilation_rate=(rate, rate), name=prefix + 'depthwise')(x)
    x = tf.keras.layers.BatchNormalization(epsilon=1e-3, momentum=0.999, name=prefix + 'depthwise_BN')(x)

    x = tf.keras.layers.Activation(tf.nn.relu6, name=prefix + 'depthwise_relu')(x)

    # Project
    x = tf.keras.layers.Conv2D(pointwise_filters, kernel_size=1, padding='same', use_bias=False, activation=None, name=prefix + 'project')(x)
    x = tf.keras.layers.BatchNormalization(epsilon=1e-3, momentum=0.999, name=prefix + 'project_BN')(x)

    if skip_connection:
        return tf.keras.layers.Add(name=prefix + 'add')([inputs, x])

    # if in_channels == pointwise_filters and stride == 1:
    #    return Add(name='res_connect_' + str(block_id))([inputs, x])

    return x

def Deeplabv3(weights='pascal_voc', input_tensor=None, input_shape=(512, 512, 3), classes=21, backbone='mobilenetv2', OS=16, alpha=1., activation=None):
    if not (weights in {'pascal_voc', 'cityscapes', None}):
        raise ValueError('The `weights` argument should be either ' '`None` (random initialization), `pascal_voc`, or `cityscapes` ' '(pre-trained on PASCAL VOC)')

    if not (backbone in {'xception', 'mobilenetv2'}):
        raise ValueError('The `backbone` argument should be either ' '`xception`  or `mobilenetv2` ')

    if input_tensor is None:
        img_input = tf.keras.Input(shape=input_shape)
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

        x = tf.keras.layers.Conv2D(32, (3, 3), strides=(2, 2), name='entry_flow_conv1_1', use_bias=False, padding='same')(img_input)

        ### Preprocessing Layer [0, 1] or [-1, 1]
        # x = tf.keras.layers.experimental.preprocessing.Rescaling(1.0 / 255)(img_input)
        # x = tf.keras.layers.experimental.preprocessing.Rescaling((1.0 / 127.5) - 1)(img_input)
        # x = tf.keras.layers.Conv2D(32, (3, 3), strides=(2, 2), name='entry_flow_conv1_1', use_bias=False, padding='same')(x)

        x = tf.keras.layers.BatchNormalization(name='entry_flow_conv1_1_BN')(x)
        x = tf.keras.layers.Activation(tf.nn.relu)(x)

        x = _conv2d_same(x, 64, 'entry_flow_conv1_2', kernel_size=3, stride=1)
        x = tf.keras.layers.BatchNormalization(name='entry_flow_conv1_2_BN')(x)
        x = tf.keras.layers.Activation(tf.nn.relu)(x)

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
        x = tf.keras.layers.Conv2D(first_block_filters, kernel_size=3, strides=(2, 2), padding='same', use_bias=False, name='Conv' if input_shape[2] == 3 else 'Conv_')(img_input)
        x = tf.keras.layers.BatchNormalization(epsilon=1e-3, momentum=0.999, name='Conv_BN')(x)
        x = tf.keras.layers.Activation(tf.nn.relu6, name='Conv_Relu6')(x)

        x = _inverted_res_block(x, filters=16, alpha=alpha, stride=1,
                                expansion=1, block_id=0, skip_connection=False)

        x = _inverted_res_block(x, filters=24, alpha=alpha, stride=2,
                                expansion=6, block_id=1, skip_connection=False)
        x = _inverted_res_block(x, filters=24, alpha=alpha, stride=1,
                                expansion=6, block_id=2, skip_connection=True)

        x = _inverted_res_block(x, filters=32, alpha=alpha, stride=2,
                                expansion=6, block_id=3, skip_connection=False)
        x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
                                expansion=6, block_id=4, skip_connection=True)
        x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
                                expansion=6, block_id=5, skip_connection=True)

        # stride in block 6 changed from 2 -> 1, so we need to use rate = 2
        x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,  # 1!
                                expansion=6, block_id=6, skip_connection=False)
        x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, rate=2,
                                expansion=6, block_id=7, skip_connection=True)
        x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, rate=2,
                                expansion=6, block_id=8, skip_connection=True)
        x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, rate=2,
                                expansion=6, block_id=9, skip_connection=True)

        x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=2,
                                expansion=6, block_id=10, skip_connection=False)
        x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=2,
                                expansion=6, block_id=11, skip_connection=True)
        x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=2,
                                expansion=6, block_id=12, skip_connection=True)

        x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, rate=2,  # 1!
                                expansion=6, block_id=13, skip_connection=False)
        x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, rate=4,
                                expansion=6, block_id=14, skip_connection=True)
        x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, rate=4,
                                expansion=6, block_id=15, skip_connection=True)

        x = _inverted_res_block(x, filters=320, alpha=alpha, stride=1, rate=4,
                                expansion=6, block_id=16, skip_connection=False)

    # end of feature extractor

    # branching for Atrous Spatial Pyramid Pooling

    # Image Feature branch
    shape_before = tf.shape(x)
    b4 = tf.keras.layers.GlobalAveragePooling2D()(x)
    b4_shape = tf.keras.backend.int_shape(b4)
    # from (b_size, channels)->(b_size, 1, 1, channels)
    b4 = tf.keras.layers.Reshape((1, 1, b4_shape[1]))(b4)
    b4 = tf.keras.layers.Conv2D(256, (1, 1), padding='same', use_bias=False, name='image_pooling')(b4)
    b4 = tf.keras.layers.BatchNormalization(name='image_pooling_BN', epsilon=1e-5)(b4)
    b4 = tf.keras.layers.Activation(tf.nn.relu)(b4)
    # upsample. have to use compat because of the option align_corners
    size_before = tf.keras.backend.int_shape(x)
    b4 = tf.keras.layers.experimental.preprocessing.Resizing(*size_before[1:3], interpolation="bilinear")(b4)
    # simple 1x1
    b0 = tf.keras.layers.Conv2D(256, (1, 1), padding='same', use_bias=False, name='aspp0')(x)
    b0 = tf.keras.layers.BatchNormalization(name='aspp0_BN', epsilon=1e-5)(b0)
    b0 = tf.keras.layers.Activation(tf.nn.relu, name='aspp0_activation')(b0)

    # there are only 2 branches in mobilenetV2. not sure why
    if backbone == 'xception':
        # rate = 6 (12)
        b1 = SepConv_BN(x, 256, 'aspp1', rate=atrous_rates[0], depth_activation=True, epsilon=1e-5)
        # rate = 12 (24)
        b2 = SepConv_BN(x, 256, 'aspp2', rate=atrous_rates[1], depth_activation=True, epsilon=1e-5)
        # rate = 18 (36)
        b3 = SepConv_BN(x, 256, 'aspp3', rate=atrous_rates[2], depth_activation=True, epsilon=1e-5)

        # concatenate ASPP branches & project
        x = tf.keras.layers.Concatenate()([b4, b0, b1, b2, b3])
    else:
        x = tf.keras.layers.Concatenate()([b4, b0])

    x = tf.keras.layers.Conv2D(256, (1, 1), padding='same', use_bias=False, name='concat_projection')(x)
    x = tf.keras.layers.BatchNormalization(name='concat_projection_BN', epsilon=1e-5)(x)
    x = tf.keras.layers.Activation(tf.nn.relu)(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    # DeepLab v.3+ decoder

    if backbone == 'xception':
        # Feature projection
        # x4 (x2) block
        skip_size = tf.keras.backend.int_shape(skip1)
        x = tf.keras.layers.experimental.preprocessing.Resizing(*skip_size[1:3], interpolation="bilinear")(x)
        dec_skip1 = tf.keras.layers.Conv2D(48, (1, 1), padding='same', use_bias=False, name='feature_projection0')(skip1)
        dec_skip1 = tf.keras.layers.BatchNormalization(name='feature_projection0_BN', epsilon=1e-5)(dec_skip1)
        dec_skip1 = tf.keras.layers.Activation(tf.nn.relu)(dec_skip1)
        x = tf.keras.layers.Concatenate()([x, dec_skip1])
        x = SepConv_BN(x, 256, 'decoder_conv0', depth_activation=True, epsilon=1e-5)
        x = SepConv_BN(x, 256, 'decoder_conv1', depth_activation=True, epsilon=1e-5)

    # you can use it with arbitary number of classes
    if (weights == 'pascal_voc' and classes == 21) or (weights == 'cityscapes' and classes == 19):
        last_layer_name = 'logits_semantic'
    else:
        last_layer_name = 'custom_logits_semantic'

    x = tf.keras.layers.Conv2D(classes, (1, 1), padding='same', name=last_layer_name)(x)
    size_before3 = tf.keras.backend.int_shape(img_input)
    x = tf.keras.layers.experimental.preprocessing.Resizing(*size_before3[1:3], interpolation="bilinear")(x)
    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = tf.keras.layers.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    if activation in {'softmax', 'sigmoid'}:
        x = tf.keras.layers.Activation(activation)(x)

    model = tf.keras.Model(inputs, x, name='deeplabv3plus')

    # load weights

    if weights == 'pascal_voc':
        if backbone == 'xception':
            weights_path = tf.keras.utils.get_file('deeplabv3_xception_tf_dim_ordering_tf_kernels.h5',
                                                   WEIGHTS_PATH_X,
                                                   cache_subdir='models')
        else:
            weights_path = tf.keras.utils.get_file('deeplabv3_mobilenetv2_tf_dim_ordering_tf_kernels.h5',
                                                   WEIGHTS_PATH_MOBILE,
                                                   cache_subdir='models')
        model.load_weights(weights_path, by_name=True)

    elif weights == 'cityscapes':
        if backbone == 'xception':
            # weights_path = tf.keras.utils.get_file('deeplabv3_xception_tf_dim_ordering_tf_kernels_cityscapes.h5',
            #                                        WEIGHTS_PATH_X_CS,
            #                                        cache_subdir='models')
            weights_path = "/data/deeplabv3_xception_tf_dim_ordering_tf_kernels_cityscapes.h5"
        else:
            weights_path = tf.keras.utils.get_file('deeplabv3_mobilenetv2_tf_dim_ordering_tf_kernels_cityscapes.h5',
                                                   WEIGHTS_PATH_MOBILE_CS,
                                                   cache_subdir='models')
        model.load_weights(weights_path, by_name=True)
    return model
