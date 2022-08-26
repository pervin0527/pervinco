import tensorflow as tf


def detector_loss(keypoint_map, logits, valid_mask):
    ## logits : 1, 15, 20, 65 --- (64 probs + 1 dustbin)
    ## img : 1, 120, 160 ---> labels : 1, 15, 20 [H / 8, W / 8]
    ## img : 1, 120, 160 ---> valid_mask : 1, 15, 20 [H / 8, W / 8]

    labels = tf.cast(keypoint_map[..., tf.newaxis], tf.float32) 
    labels = tf.nn.space_to_depth(labels, 8)

    shape = tf.concat([tf.shape(labels)[:3], [1]], axis=0)
    labels = tf.concat([2*labels, tf.ones(shape)], 3)
    labels = tf.argmax(labels + tf.random.uniform(tf.shape(labels), 0, 0.1), axis=3)
    # labels = tf.argmax(labels, axis=3)

    valid_mask = tf.ones_like(keypoint_map) if valid_mask is None else valid_mask
    valid_mask = tf.cast(valid_mask[..., tf.newaxis], tf.float32)
    valid_mask = tf.nn.space_to_depth(valid_mask, 8)
    valid_mask = tf.reduce_prod(valid_mask, axis=3)

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels) ## 1, 15, 20
    loss = tf.math.divide_no_nan(tf.reduce_sum(loss * valid_mask), tf.reduce_sum(valid_mask))

    return loss