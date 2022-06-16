import hparams as param
import tensorflow as tf

def loss_fn(attribute_gt, landmark_gt, euler_angle_gt, angle, landmarks):
    weight_angle = tf.reduce_sum(1 - tf.cos(angle - euler_angle_gt), axis=1)
    attributes_w_n = attribute_gt[:, 1:6]
    # attributes_w_n = tf.where(attributes_w_n > 0, attributes_w_n, 0.1 / cfg.BATCH_SIZE)

    mat_ratio = tf.reduce_mean(attributes_w_n, axis=0)
    mat_ratio = tf.where(mat_ratio > 0, 1.0 / mat_ratio, param.BATCH_SIZE)

    weight_attribute = tf.reduce_sum(tf.multiply(attributes_w_n, mat_ratio), axis=1)
    l2_distance = tf.reduce_sum((landmark_gt - landmarks) * (landmark_gt - landmarks), axis=1)

    weighted_loss = tf.reduce_mean(weight_angle * weight_attribute * l2_distance)
    loss = tf.reduce_mean(l2_distance)

    return weighted_loss, loss


def wing_loss(y_true, y_pred, w=10.0, epsilon=2.0, N_LANDMARK=98):
    y_pred = tf.reshape(y_pred, (-1, N_LANDMARK, 2))
    y_true = tf.reshape(y_true, (-1, N_LANDMARK, 2))

    x = y_true - y_pred
    c = w * (1.0 - tf.math.log(1.0 + w / epsilon))
    absolute_x = tf.abs(x)
    losses = tf.where(w > absolute_x,
                      w * tf.math.log(1.0 + absolute_x / epsilon),
                      absolute_x - c)
    loss = tf.reduce_mean(tf.reduce_sum(losses, axis=[1, 2]), axis=0)
    return loss