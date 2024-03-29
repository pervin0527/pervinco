import sys
from xml.etree.ElementInclude import include
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import AveragePooling2D, Lambda, Conv2D, Conv2DTranspose, Activation, Reshape, concatenate, Concatenate, BatchNormalization, ZeroPadding2D


def Upsample(tensor, size):
    name = tensor.name.split('/')[0] + '_upsample'

    def bilinear_upsample(x, size):
        resized = tf.image.resize(images=x, size=size)
        return resized
    y = Lambda(lambda x: bilinear_upsample(x, size), output_shape=size, name=name)(tensor)
    return y


def ASPP(tensor):
    dims = K.int_shape(tensor)

    y_pool = AveragePooling2D(pool_size=(dims[1], dims[2]), name='average_pooling')(tensor)
    y_pool = Conv2D(filters=256, kernel_size=1, padding='same', kernel_initializer='he_normal', name='pool_1x1conv2d', use_bias=False)(y_pool)
    y_pool = BatchNormalization(name=f'bn_1')(y_pool)
    y_pool = Activation('relu', name=f'relu_1')(y_pool)

    y_pool = Upsample(tensor=y_pool, size=[dims[1], dims[2]])

    y_1 = Conv2D(filters=256, kernel_size=1, dilation_rate=1, padding='same', kernel_initializer='he_normal', name='ASPP_conv2d_d1', use_bias=False)(tensor)
    y_1 = BatchNormalization(name=f'bn_2')(y_1)
    y_1 = Activation('relu', name=f'relu_2')(y_1)

    y_6 = Conv2D(filters=256, kernel_size=3, dilation_rate=6, padding='same', kernel_initializer='he_normal', name='ASPP_conv2d_d6', use_bias=False)(tensor)
    y_6 = BatchNormalization(name=f'bn_3')(y_6)
    y_6 = Activation('relu', name=f'relu_3')(y_6)

    y_12 = Conv2D(filters=256, kernel_size=3, dilation_rate=12, padding='same', kernel_initializer='he_normal', name='ASPP_conv2d_d12', use_bias=False)(tensor)
    y_12 = BatchNormalization(name=f'bn_4')(y_12)
    y_12 = Activation('relu', name=f'relu_4')(y_12)

    y_18 = Conv2D(filters=256, kernel_size=3, dilation_rate=18, padding='same', kernel_initializer='he_normal', name='ASPP_conv2d_d18', use_bias=False)(tensor)
    y_18 = BatchNormalization(name=f'bn_5')(y_18)
    y_18 = Activation('relu', name=f'relu_5')(y_18)

    y = concatenate([y_pool, y_1, y_6, y_12, y_18], name='ASPP_concat')

    y = Conv2D(filters=256, kernel_size=1, dilation_rate=1, padding='same', kernel_initializer='he_normal', name='ASPP_conv2d_final', use_bias=False)(y)
    y = BatchNormalization(name=f'bn_final')(y)
    y = Activation('relu', name=f'relu_final')(y)
    return y


def DeepLabV3Plus(img_height, img_width, nclasses=66, backbone_name="resnet50", backbone_trainable=False, final_activation=None):
    model_input = tf.keras.Input(shape=(img_width, img_height, 3))
    # model_input = tf.keras.layers.experimental.preprocessing.Rescaling(1.0 / 255.0)(model_input)
    # model_input = tf.keras.layers.experimental.preprocessing.Rescaling((1.0 / 127.5) - 1)(model_input)
    
    if backbone_name.lower() == "resnet50":
        base_model = tf.keras.applications.ResNet50(input_tensor=model_input, weights='imagenet', include_top=False)
                
        layer_names = ["conv4_block6_2_relu", "conv2_block3_2_relu"]
        upsample_scale = [(img_height // 4), (img_width // 4)]

    elif backbone_name.lower() == "resnet101":
        base_model = tf.keras.applications.ResNet101(input_tensor=model_input, weights='imagenet', include_top=False)

        layer_names = ["conv4_block23_1_relu", "conv2_block3_2_relu"]
        upsample_scale = [(img_height // 4), (img_width // 4)]

    elif backbone_name.lower() == "xception":
        base_model = tf.keras.applications.Xception(input_tensor=model_input, weights='imagenet', include_top=False)
        
        # layer_names = ["block14_sepconv2_act", "block3_sepconv2_act"]
        # layer_names = ["block4_sepconv2_act", "block3_sepconv2_act"]
        layer_names = ["block13_sepconv2_act", "block3_sepconv2_act"]
        
        upsample_scale = [(img_height // 4) - 1, (img_width // 4) - 1]

    elif backbone_name.lower() == "efficientnetb0":
        base_model = tf.keras.applications.EfficientNetB0(input_tensor=model_input, weights="imagenet", include_top=False)
        # ["block2b_activation", "block6d_activation"]
        layer_names = ["block6a_expand_activation", "block3a_expand_activation"]
        upsample_scale = [(img_height // 4), (img_width // 4)]

    elif backbone_name.lower() == "efficientnetb3":
        base_model = tf.keras.applications.EfficientNetB3(input_tensor=model_input, weights="imagenet", include_top=False)

        # ["block2a_expand_activation", "block3a_expand_activation", "block4a_expand_activation", "block6a_expand_activation", "top_activation"]
        layer_names = ["block6a_expand_activation", "block3a_expand_activation"]
        upsample_scale = [(img_height // 4), (img_width // 4)]

    base_model.trainable = backbone_trainable
    
    image_features = base_model.get_layer(layer_names[0]).output
    x_a = ASPP(image_features)
    x_a = Upsample(tensor=x_a, size=[upsample_scale[0], upsample_scale[1]])

    x_b = base_model.get_layer(layer_names[1]).output
    x_b = Conv2D(filters=48, kernel_size=1, padding='same', kernel_initializer='he_normal', name='low_level_projection', use_bias=False)(x_b)
    x_b = BatchNormalization(name=f'bn_low_level_projection')(x_b)
    x_b = Activation('relu', name='low_level_activation')(x_b)

    # print(x_a.shape, x_b.shape)
    x = concatenate([x_a, x_b], name='decoder_concat')
    # sys.exit()

    x = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal', name='decoder_conv2d_1', use_bias=False)(x)
    x = BatchNormalization(name=f'bn_decoder_1')(x)
    x = Activation('relu', name='activation_decoder_1')(x)

    x = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal', name='decoder_conv2d_2', use_bias=False)(x)
    x = BatchNormalization(name=f'bn_decoder_2')(x)
    x = Activation('relu', name='activation_decoder_2')(x)

    x = Conv2D(nclasses, (1, 1), name='output_layer')(x)
    x = Upsample(x, [x.shape[1], x.shape[2]])
    x = Upsample(x, [img_height, img_width])

    if final_activation != None:
        x = Activation(final_activation)(x)

    model = Model(inputs=base_model.input, outputs=x, name='DeepLabV3_Plus')
    return model