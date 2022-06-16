import tensorflow as tf

def loss_fn(attribute_gt, landmark_gt, euler_angle_gt, angle, landmarks):
    batch_size = 64
    weight_angle = tf.reduce_sum(1 - tf.cos(angle - euler_angle_gt), axis=1)
    attributes_w_n = attribute_gt[:, 1:6]
    # attributes_w_n = tf.where(attributes_w_n > 0, attributes_w_n, 0.1 / cfg.BATCH_SIZE)

    mat_ratio = tf.reduce_mean(attributes_w_n, axis=0)
    mat_ratio = tf.where(mat_ratio > 0, 1.0 / mat_ratio, batch_size)

    weight_attribute = tf.reduce_sum(tf.multiply(attributes_w_n, mat_ratio), axis=1)

    l2_distance = tf.reduce_sum((landmark_gt - landmarks) * (landmark_gt - landmarks), axis=1)

    weighted_loss = tf.reduce_mean(weight_angle * weight_attribute * l2_distance)

    loss = tf.reduce_mean(l2_distance)

    return weighted_loss, loss