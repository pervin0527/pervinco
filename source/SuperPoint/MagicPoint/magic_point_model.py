import itertools
import tensorflow as tf
from losses import detector_loss
from tensorflow.keras import backend as K

def basic_block(inputs, filters, kernel_size, strides, padding="SAME", activation="relu"):
    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=(kernel_size, kernel_size), strides=strides, padding=padding)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation)(x)

    return x


def resnet_backbone(inputs):
    inputs = tf.keras.Input(shape=inputs) ## 120, 160, 1
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding="SAME")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = basic_block(x, 64, 3, 1, "SAME", "relu")
    x = basic_block(x, 64, 3, 1, "SAME", "relu")
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding="SAME")(x) ## 60, 80, 64

    x = basic_block(x, 64, 3, 1, "SAME", "relu")
    x = basic_block(x, 64, 3, 1, "SAME", "relu")
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding="SAME")(x) ## 30, 40, 64

    x = basic_block(x, 128, 3, 1, "SAME", "relu")
    x = basic_block(x, 128, 3, 1, "SAME", "relu")
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding="SAME")(x) ## 15, 20, 128

    x = basic_block(x, 128, 3, 1, "SAME", "relu")
    x = basic_block(x, 128, 3, 1, "SAME", "relu")

    model = tf.keras.Model(inputs=inputs, outputs=x)
    return model
    

def vgg_block(inputs, filters, kernel_size, activation):
    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, padding="SAME", kernel_regularizer=tf.keras.regularizers.L2(0.))(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation)(x)

    return x


def vgg_backbone(inputs):
    inputs = tf.keras.Input(shape=inputs)
    x = vgg_block(inputs, 64, 3, "relu") ## 120, 160, 64
    x = vgg_block(x, 64, 3, "relu") ## 120, 160, 64
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding="SAME")(x) ## 60, 80, 64

    x = vgg_block(x, 64, 3, "relu") ## 60, 80, 64
    x = vgg_block(x, 64, 3, "relu") ## 60, 80, 64
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding="SAME")(x) ## 30, 40, 64

    x = vgg_block(x, 128, 3, "relu") ## 30, 40, 128
    x = vgg_block(x, 128, 3, "relu") ## 30, 40, 128
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding="SAME")(x) ## 15, 20, 128

    x = vgg_block(x, 128, 3, "relu")
    output = vgg_block(x, 128, 3, "relu")

    model = tf.keras.Model(inputs=inputs, outputs=output)
    return model


def detector_head(inputs):
    inputs = tf.keras.Input(shape=inputs)
    x = vgg_block(inputs, 256, 3, "relu")
    x = vgg_block(x, 1 + pow(8, 2), 1, None)

    prob = tf.keras.activations.softmax(x, axis=-1)
    prob = prob[:, :, :, :-1]
    prob = tf.nn.depth_to_space(prob, 8, data_format='NHWC')
    prob = tf.squeeze(prob, axis=-1)

    model = tf.keras.Model(inputs=inputs, outputs=[x, prob])

    return model


class MagicPoint(tf.keras.Model):
    def __init__(self, backbone_input, summary=False):
        super(MagicPoint, self).__init__()
        # self.vgg_backbone = vgg_backbone(inputs=(backbone_input))
        self.resnet_backbone = resnet_backbone(inputs=(backbone_input))
        self.detector_head = detector_head(inputs=(int(backbone_input[0] / 8), int(backbone_input[1] / 8), 128))
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

        if summary:
            # self.vgg_backbone.summary()
            self.resnet_backbone.summary()
            self.detector_head.summary()

    def call(self, x, training=False):
        # backbone_output = self.vgg_backbone(x)
        backbone_output = self.resnet_backbone(x)
        logits, prob = self.detector_head(backbone_output)

        return logits, prob

    def train_step(self, data):
        image, keypoints, valid_mask, keypoint_map = data["image"], data["keypoints"], data["valid_mask"], data["keypoint_map"]

        with tf.GradientTape() as tape:
            pred_logits, pred_prob = self(image, training=True)
            loss = detector_loss(keypoint_map, pred_logits, valid_mask)
            # loss = detector_loss(keypoint_map, pred_prob, valid_mask)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.loss_tracker.update_state(loss)

        return {"loss" : self.loss_tracker.result()}

    def test_step(self, data):
        image, keypoints, valid_mask, keypoint_map = data["image"], data["keypoints"], data["valid_mask"], data["keypoint_map"]
        pred_logits, pred_prob = self(image, training=False)
        loss = detector_loss(keypoint_map, pred_logits, valid_mask)
        # loss = detector_loss(keypoint_map, pred_prob, valid_mask)
        
        self.loss_tracker.update_state(loss)
        return {"loss" : self.loss_tracker.result()}