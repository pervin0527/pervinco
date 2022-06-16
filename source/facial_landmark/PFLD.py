import losses
import tensorflow as tf


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


def backbone(input_size, n_landmarks):
    input_layer = tf.keras.layers.Input(shape=(input_size, input_size, 3))
    
    ## 112, 112, 3 ----> 56, 56, 64
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, padding="same")(input_layer)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    ## 56, 56, 64 ----> 56, 56, 64
    # x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, padding="same")(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    ## 56, 56, 64 ----> 28, 28, 64
    ## repeated 5, stride first layer : 2, expansion : 2
    x = inverted_residual_block(x, expansion=2, stride=2, filters=64, use_res_connect=False)
    x = inverted_residual_block(x, expansion=2, stride=1, filters=64, use_res_connect=True)
    x = inverted_residual_block(x, expansion=2, stride=1, filters=64, use_res_connect=True)
    x = inverted_residual_block(x, expansion=2, stride=1, filters=64, use_res_connect=True)
    out1 = inverted_residual_block(x, expansion=2, stride=1, filters=64, use_res_connect=True) ## 28, 28, 64

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
    output_layer = tf.keras.layers.Dense(n_landmarks)(multi_scale)

    model = tf.keras.Model(inputs=input_layer, outputs=[out1, output_layer])
    return model


def auxiliarynet(input_size):
    input_layer = tf.keras.layers.Input(shape=(input_size, input_size, 64)) ## 28, 28, 64

    ## 28, 28, 64 ---> 14, 14, 128
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=2, padding="same")(input_layer)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    ## 14, 14, 128 ---> 14, 14, 128
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    ## 14, 14, 128 ---> 7, 7, 32
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=2, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    ## 7, 7, 32 ----> 3, 3, 128
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=7, strides=3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.MaxPool2D(pool_size=3, padding="same")(x) ## 1, 1, 128
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(32)(x)
    output_layer = tf.keras.layers.Dense(3)(x)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    return model


class PFLD(tf.keras.Model):
    def __init__(self, input_size=112, n_landmarks=98 * 2, summary=False):
        super(PFLD, self).__init__()
        self.pfld_model = backbone(input_size, n_landmarks)
        self.auxiliary_model = auxiliarynet(input_size=int(input_size / 4))

        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.loss_tracker_2 = tf.keras.metrics.Mean(name="weighted_loss")

        if summary:
            self.pfld_model.summary()
            self.auxiliary_model.summary()

    def call(self, x, training=False):
        features, landmark = self.pfld_model(x)
        if training:
            angle = self.auxiliary_model(features)
            return angle, landmark
        else:
            return landmark

    def train_step(self, data):
        img_tensor, attribute_gt, landmark_gt, euler_angle_gt = data
        
        with tf.GradientTape() as tape:
            angle, landmarks = self(img_tensor, training=True)
            weighted_loss, loss = losses.loss_fn(attribute_gt, landmark_gt, euler_angle_gt, angle, landmarks)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(weighted_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.loss_tracker.update_state(loss)
        self.loss_tracker_2.update_state(weighted_loss)
        
        return {"loss": self.loss_tracker.result(), "weighted_loss": self.loss_tracker_2.result()}

    @property
    def metrics(self):
        return [self.loss_tracker, self.loss_tracker_2]