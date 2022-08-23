import itertools
import tensorflow as tf


def detector_loss(keypoint_map, logits, valid_mask, is_focal):
    labels = tf.cast(keypoint_map[..., tf.newaxis], tf.float32) 
    labels = tf.nn.space_to_depth(labels, 8)

    shape = tf.concat([tf.shape(labels)[:3], [1]], axis=0)
    labels = tf.concat([2*labels, tf.ones(shape)], 3)
    
    labels = tf.argmax(labels + tf.random.uniform(tf.shape(labels), 0, 0.1), axis=3)

    valid_mask = tf.ones_like(keypoint_map) if valid_mask is None else valid_mask
    valid_mask = tf.cast(valid_mask[..., tf.newaxis], tf.float32)
    valid_mask = tf.nn.space_to_depth(valid_mask, 8)
    valid_mask = tf.reduce_prod(valid_mask, axis=3)

    # tf.print(tf.shape(valid_mask), tf.shape(labels), tf.shape(logits))
    ## valid_mask : 1, 15, 20
    ## labels : 1, 15, 20
    ## logits : 1, 15, 20 , 65

    if not is_focal:
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        loss *= valid_mask
    else:
        loss = sparse_categorical_focal_loss(y_true=labels, y_pred=logits, gamma=2, alpha=0.25, from_logits=True)

    return loss


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

    if reshape_needed:
        loss = tf.reshape(loss, y_pred_shape[:-1])

    return loss