import itertools
import tensorflow as tf

from typing import Any, Optional

class SparseCategoricalFocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma, class_weight: Optional[Any] = None, from_logits: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.class_weight = class_weight
        self.from_logits = from_logits

    def get_config(self):
        config = super().get_config()
        config.update(gamma=self.gamma, class_weight=self.class_weight, from_logits=self.from_logits)
        return config

    def call(self, y_true, y_pred):
        return sparse_categorical_focal_loss(y_true=y_true, y_pred=y_pred, class_weight=self.class_weight, gamma=self.gamma, from_logits=self.from_logits)


def sparse_categorical_focal_loss(y_true, y_pred, gamma, *, class_weight: Optional[Any] = None, from_logits: bool = False, axis: int = -1) -> tf.Tensor:
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

    focal_modulation = (1 - probs) ** gamma
    loss = focal_modulation * xent_loss

    if class_weight is not None:
        class_weight = tf.gather(class_weight, y_true, axis=0, batch_dims=y_true_rank)
        loss *= class_weight
        # loss *= 0.25

    if reshape_needed:
        loss = tf.reshape(loss, y_pred_shape[:-1])

    return loss


def categorical_focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.keras.backend.clip(y_pred, epsilon, 1.0-epsilon)
        cross_entropy = -y_true*tf.keras.backend.log(y_pred)
        weight = alpha * y_true * tf.keras.backend.pow((1-y_pred), gamma)
        loss = weight * cross_entropy
        loss = tf.keras.backend.sum(loss, axis=1)
        return loss
    
    return focal_loss                                            