import tensorflow as tf
from tensorflow.keras import backend as K

def vgg_block(inputs, filters, kernel_size, activation):
    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, padding="SAME", kernel_regularizer=tf.keras.regularizers.L2(0.))(inputs)
    x = tf.keras.layers.Activation(activation)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    return x


def vgg_backbone(inputs):
    inputs = tf.keras.Input(shape=inputs)
    x = vgg_block(inputs, 64, 3, "relu")
    x = vgg_block(x, 64, 3, "relu")
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding="SAME")(x)

    x = vgg_block(x, 64, 3, "relu")
    x = vgg_block(x, 64, 3, "relu")
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding="SAME")(x)

    x = vgg_block(x, 128, 3, "relu")
    x = vgg_block(x, 128, 3, "relu")
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding="SAME")(x)

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


def detector_loss(keypoint_map, logits, valid_mask):
    ### keypoint_map : 120, 160
    ### logits : 15, 20, 65
    ### valid_mask : 120, 160
    
    labels = tf.cast(keypoint_map[..., tf.newaxis], tf.float32) 
    labels = tf.nn.space_to_depth(labels, 8)

    shape = tf.concat([tf.shape(labels)[:3], [1]], axis=0)
    labels = tf.concat([2*labels, tf.ones(shape)], 3)
    
    labels = tf.argmax(labels + tf.random.uniform(tf.shape(labels), 0, 0.1), axis=3)

    valid_mask = tf.ones_like(keypoint_map) if valid_mask is None else valid_mask
    valid_mask = tf.cast(valid_mask[..., tf.newaxis], tf.float32)
    valid_mask = tf.nn.space_to_depth(valid_mask, 8)
    valid_mask = tf.reduce_prod(valid_mask, axis=3)

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    loss = K.mean(tf.math.multiply(loss, valid_mask))

    # keypoint_map = tf.cast(keypoint_map, tf.float32)
    # logits = tf.cast(logits, tf.float32)
    # valid_mask = tf.cast(valid_mask, tf.float32)
    # loss = tf.nn.weighted_cross_entropy_with_logits(logits=logits, labels=keypoint_map, pos_weight=valid_mask)

    return loss


class MagicPoint(tf.keras.Model):
    def __init__(self, backbone_input, summary=False):
        super(MagicPoint, self).__init__()
        self.vgg_backbone = vgg_backbone(inputs=(backbone_input))
        self.detector_head = detector_head(inputs=(int(backbone_input[0] / 8), int(backbone_input[1] / 8), 128))
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

        if summary:
            self.vgg_backbone.summary()
            self.detector_head.summary()

    def call(self, x, training=False):
        backbone_output = self.vgg_backbone(x)
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
            

if __name__ == "__main__":
    input_shape = (160, 120, 1) ## Wc = W / 8, Hc = H / 8 ------> 20, 15
    magic_point = vgg_backbone(input_shape)
    magic_point.summary()

    detector_head_model = detector_head((int(input_shape[0]/8), int(input_shape[1]/8), 128))
    detector_head_model.summary()