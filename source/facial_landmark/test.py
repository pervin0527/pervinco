import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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

def inverted_residual_block(input, expansion, stride, filters, use_res_connect):
    input_channel = tf.keras.backend.int_shape(input)[-1]
    assert stride in [1, 2]

    block = tf.keras.Sequential([
    tf.keras.layers.Conv2D(expansion * input_channel, kernel_size=1, padding="valid"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Conv2D(expansion * input_channel, kernel_size=3, strides=stride, padding="same"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Conv2D(filters=filters, kernel_size=1, strides=1, padding="valid"),
    tf.keras.layers.BatchNormalization(),
    ])

    if use_res_connect:
        # return tf.keras.layers.Add()([input, x])
        return input + block(input)

    return block(input)


def backbone():
    input_layer = tf.keras.layers.Input(shape=(112, 112, 3))
    
    ## 112, 112, 3 ----> 56, 56, 64
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, padding="same")(input_layer)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    ## 56, 56, 64 ----> 56, 56, 64
    # x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, padding="same")(x)
    x = tf.keras.layers.Conv2d(filters=64, kernel_size=3, strides=1, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    ## 56, 56, 64 ----> 28, 28, 64
    ## repeated 5, stride first layer : 2, expansion : 2
    x = inverted_residual_block(x, expansion=2, stride=2, filters=64, use_res_connect=False)
    x = inverted_residual_block(x, expansion=2, stride=1, filters=64, use_res_connect=True)
    x = inverted_residual_block(x, expansion=2, stride=1, filters=64, use_res_connect=True)
    x = inverted_residual_block(x, expansion=2, stride=1, filters=64, use_res_connect=True)
    out1 = inverted_residual_block(x, expansion=2, stride=1, filters=64, use_res_connect=True)

    ## 28, 28, 64 ----> 14, 14, 128
    x = inverted_residual_block(out1, expansion=2, stride=2, filters=128, use_res_connect=False)

    ## 14, 14, 128 ----> 14, 14, 128
    x = inverted_residual_block(x, expansion=4, stride=1, filters=128, use_res_connect=False)
    x = inverted_residual_block(x, expansion=4, stride=1, filters=128, use_res_connect=True)
    x = inverted_residual_block(x, expansion=4, stride=1, filters=128, use_res_connect=True)
    x = inverted_residual_block(x, expansion=4, stride=1, filters=128, use_res_connect=True)
    x = inverted_residual_block(x, expansion=4, stride=1, filters=128, use_res_connect=True)
    x = inverted_residual_block(x, expansion=4, stride=1, filters=128, use_res_connect=True)

    ## 14, 14, 128 ----> 14, 14, 16
    x = inverted_residual_block(x, expansion=2, stride=1, filters=16, use_res_connect=False)

    ## S1 : 14, 14, 16
    s1 = tf.keras.layers.AvgPool2D(14)(x)
    s1 = tf.keras.layers.Flatten()(s1)
    
    ## S2 : 14, 14, 16 ----> 7, 7, 32
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=2, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    s2 = tf.keras.layers.AvgPool2D(7)(x)
    s2 = tf.keras.layers.Flatten()(s2)

    ## S3 : 7, 7, 32 ----> 1, 1, 128
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=7, strides=1, padding="valid")(x)
    x = tf.keras.layers.ReLU()(x)

    s3 = tf.keras.layers.Flatten()(x)

    multi_scale = tf.keras.layers.concatenate([s1, s2, s3], axis=1)
    output_layer = tf.keras.layers.Dense(196)(multi_scale)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    model.summary()

    return model

model = backbone()
model.summary