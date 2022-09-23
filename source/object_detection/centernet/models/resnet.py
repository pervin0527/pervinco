import tensorflow as tf

def identity_block(input_tensor, kernel_size, filters, stage, block):
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = tf.keras.layers.Conv2D(filters1, (1, 1), kernel_initializer=tf.keras.initializers.random_normal(stddev=0.02), name=conv_name_base + '2a', use_bias=False)(input_tensor)
    x = tf.keras.layers.BatchNormalization(name=bn_name_base + '2a')(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2D(filters2, kernel_size,padding='same', kernel_initializer=tf.keras.initializers.random_normal(stddev=0.02), name=conv_name_base + '2b', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(name=bn_name_base + '2b')(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2D(filters3, (1, 1), kernel_initializer=tf.keras.initializers.random_normal(stddev=0.02), name=conv_name_base + '2c', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(name=bn_name_base + '2c')(x)

    x = tf.keras.layers.add([x, input_tensor])
    x = tf.keras.layers.Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = tf.keras.layers.Conv2D(filters1, (1, 1), strides=strides, kernel_initializer=tf.keras.initializers.random_normal(stddev=0.02), name=conv_name_base + '2a', use_bias=False)(input_tensor)
    x = tf.keras.layers.BatchNormalization(name=bn_name_base + '2a')(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2D(filters2, kernel_size, padding='same', kernel_initializer=tf.keras.initializers.random_normal(stddev=0.02), name=conv_name_base + '2b', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(name=bn_name_base + '2b')(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2D(filters3, (1, 1), kernel_initializer=tf.keras.initializers.random_normal(stddev=0.02), name=conv_name_base + '2c', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(name=bn_name_base + '2c')(x)

    shortcut = tf.keras.layers.Conv2D(filters3, (1, 1), strides=strides, kernel_initializer=tf.keras.initializers.random_normal(stddev=0.02), name=conv_name_base + '1', use_bias=False)(input_tensor)
    shortcut = tf.keras.layers.BatchNormalization(name=bn_name_base + '1')(shortcut)

    x = tf.keras.layers.add([x, shortcut])
    x = tf.keras.layers.Activation('relu')(x)
    return x


def ResNet50(inputs):
    # 512x512x3
    x = tf.keras.layers.ZeroPadding2D((3, 3))(inputs)
    # 256,256,64
    x = tf.keras.layers.Conv2D(64, (7, 7), strides=(2, 2), kernel_initializer=tf.keras.initializers.random_normal(stddev=0.02), name='conv1', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(name='bn_conv1')(x)
    x = tf.keras.layers.Activation('relu')(x)

    # 256,256,64 -> 128,128,64
    x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)

    # 128,128,64 -> 128,128,256
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    # 128,128,256 -> 64,64,512
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    # 64,64,512 -> 32,32,1024
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    # 32,32,1024 -> 16,16,2048
    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    return x

def centernet_head(x,num_classes):
    x = tf.keras.layers.Dropout(rate=0.5)(x)
    num_filters = 256
    # 16, 16, 2048  ->  32, 32, 256 -> 64, 64, 128 -> 128, 128, 64
    for i in range(3):
        x = tf.keras.layers.Conv2DTranspose(num_filters // pow(2, i), (4, 4), strides=2, use_bias=False, padding='same', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.L2(5e-4))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)

    # hm header
    y1 = tf.keras.layers.Conv2D(64, 3, padding='same', use_bias=False, kernel_initializer=tf.keras.initializers.random_normal(stddev=0.02))(x)
    y1 = tf.keras.layers.BatchNormalization()(y1)
    y1 = tf.keras.layers.Activation('relu')(y1)
    y1 = tf.keras.layers.Conv2D(num_classes, 1, kernel_initializer=tf.keras.initializers.constant(0), bias_initializer=tf.keras.initializers.constant(-2.19), activation='sigmoid')(y1)

    # wh header
    y2 = tf.keras.layers.Conv2D(64, 3, padding='same', use_bias=False, kernel_initializer=tf.keras.initializers.random_normal(stddev=0.02))(x)
    y2 = tf.keras.layers.BatchNormalization()(y2)
    y2 = tf.keras.layers.Activation('relu')(y2)
    y2 = tf.keras.layers.Conv2D(2, 1, kernel_initializer=tf.keras.initializers.random_normal(stddev=0.02))(y2)

    # reg header
    y3 = tf.keras.layers.Conv2D(64, 3, padding='same', use_bias=False, kernel_initializer=tf.keras.initializers.random_normal(stddev=0.02))(x)
    y3 = tf.keras.layers.BatchNormalization()(y3)
    y3 = tf.keras.layers.Activation('relu')(y3)
    y3 = tf.keras.layers.Conv2D(2, 1, kernel_initializer=tf.keras.initializers.random_normal(stddev=0.02))(y3)
    
    return y1, y2, y3
