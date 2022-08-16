import itertools
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


def sparse_categorical_focal_loss(y_true, y_pred, gamma, alpha, class_weight = None, from_logits = False, axis= -1):
    gamma = tf.convert_to_tensor(gamma, dtype=tf.dtypes.float32)
    gamma_rank = gamma.shape.rank
    scalar_gamma = gamma_rank == 0

    if class_weight is not None:
        class_weight = tf.convert_to_tensor(class_weight, dtype=tf.dtypes.float32)

    y_pred = tf.convert_to_tensor(y_pred)
    y_pred_rank = y_pred.shape.rank

    if y_pred_rank is not None:
        axis %= y_pred_rank
        if axis != y_pred_rank - 1:
            perm = list(itertools.chain(range(axis), range(axis + 1, y_pred_rank), [axis]))
            y_pred = tf.transpose(y_pred, perm=perm)

    elif axis != -1:
        raise ValueError(f'Cannot compute sparse categorical focal loss with axis={axis} on a prediction tensor with statically unknown rank.')

    y_pred_shape = tf.shape(y_pred)
    y_true = tf.dtypes.cast(y_true, dtype=tf.dtypes.int64)
    y_true_rank = y_true.shape.rank

    if y_true_rank is None:
        raise NotImplementedError('Sparse categorical focal loss not supported for target/label tensors of unknown rank')

    reshape_needed = (y_true_rank is not None and y_pred_rank is not None and y_pred_rank != y_true_rank + 1)
    if reshape_needed:
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1, y_pred_shape[-1]])

    if from_logits:
        logits = y_pred
        probs = tf.nn.softmax(y_pred, axis=-1)
    else:
        probs = y_pred
        logits = tf.math.log(tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon()))

    xent_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=logits,)

    y_true_rank = y_true.shape.rank
    probs = tf.gather(probs, y_true, axis=-1, batch_dims=y_true_rank)

    if not scalar_gamma:
        gamma = tf.gather(gamma, y_true, axis=0, batch_dims=y_true_rank)

    # focal_modulation = (1 - probs) ** gamma
    # loss = focal_modulation * xent_loss
    loss = alpha * (1 - probs) ** gamma * xent_loss

    if class_weight is not None:
        class_weight = tf.gather(class_weight, y_true, axis=0, batch_dims=y_true_rank)
        loss *= class_weight
        # loss *= 0.25

    if reshape_needed:
        loss = tf.reshape(loss, y_pred_shape[:-1])

    return loss


def detector_loss(keypoint_map, logits, valid_mask):
    labels = tf.cast(keypoint_map[..., tf.newaxis], tf.float32) 
    labels = tf.nn.space_to_depth(labels, 8)

    shape = tf.concat([tf.shape(labels)[:3], [1]], axis=0)
    labels = tf.concat([2*labels, tf.ones(shape)], 3)
    
    labels = tf.argmax(labels + tf.random.uniform(tf.shape(labels), 0, 0.1), axis=3)

    valid_mask = tf.ones_like(keypoint_map) if valid_mask is None else valid_mask
    valid_mask = tf.cast(valid_mask[..., tf.newaxis], tf.float32)
    valid_mask = tf.nn.space_to_depth(valid_mask, 8)
    valid_mask = tf.reduce_prod(valid_mask, axis=3)

    # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    loss = sparse_categorical_focal_loss(labels, logits, gamma=1.5, alpha=0.25, class_weight=None, from_logits=True)

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