import math
import tensorflow as tf
from tensorflow.keras import backend as K

def PFLDLoss(label, landmark_pred, angle_pred):
    batch_size = tf.cast(K.shape(label)[0], tf.float32)
    landmark_gt, attribute_gt, euler_angle_gt = tf.cast(label[:, :136], tf.float32), tf.cast(label[:, 136:142], tf.float32), tf.cast(label[:, 142:], tf.float32)
    
    weight_angle = K.sum(1 - tf.cos(angle_pred - euler_angle_gt), axis=1)
    attributes_w_n = tf.cast(attribute_gt[:, 1:6], tf.float32)

    # mat_ratio = K.mean(attributes_w_n, axis=0)
    # N = K.shape(mat_ratio)[0]
    # mat_ratio = tf.where(mat_ratio>0, 1.0/mat_ratio, batch_size)
    # weight_attribute = K.sum(tf.matmul(attributes_w_n, K.reshape(mat_ratio, (N,1))), axis=1) # [8,1]

    # l2_distant = K.sum((landmark_gt - landmark_pred) * (landmark_gt - landmark_pred), axis=1)
    # loss = tf.reduce_mean(l2_distant)
    # weighted_loss = tf.reduce_mean(weight_angle * weight_attribute * l2_distant)

    mat_ratio = K.mean(attributes_w_n, axis=0)
    mat_ratio = tf.where(mat_ratio > 0, 1.0 / mat_ratio, batch_size)
    weight_attribute = K.sum(tf.multiply(attributes_w_n, mat_ratio), 1)

    l2_distant = K.sum((landmark_gt - landmark_pred) * (landmark_gt - landmark_pred), axis=1)
    loss = K.mean(l2_distant)
    weighted_loss = K.mean(weight_angle * weight_attribute * l2_distant)


    return weighted_loss, loss


### Epoch 94 : loss: 0.0623 - weighted_loss: 0.5965 - val_loss: 30.1277 - lr: 7.8104e-04
def loss_fn(label, landmark_pred, angle_pred):
    batch_size = tf.cast(K.shape(label)[0], tf.float32)
    landmark_gt, attribute_gt, euler_angle_gt = tf.cast(label[:, :136], tf.float32), tf.cast(label[:, 136:142], tf.float32), tf.cast(label[:, 142:], tf.float32)

    weight_angle = tf.reduce_sum(1 - tf.cos(angle_pred - euler_angle_gt), axis=1)
    attributes_w_n = attribute_gt[:, 1:6]
    # attributes_w_n = tf.where(attributes_w_n > 0, attributes_w_n, 0.1 / cfg.BATCH_SIZE)

    mat_ratio = tf.reduce_mean(attributes_w_n, axis=0)
    mat_ratio = tf.where(mat_ratio > 0, 1.0 / mat_ratio, batch_size)

    weight_attribute = tf.reduce_sum(tf.multiply(attributes_w_n, mat_ratio), axis=1)

    l2_distance = tf.reduce_sum((landmark_gt - landmark_pred) * (landmark_gt - landmark_pred), axis=1)

    weighted_loss = tf.reduce_mean(weight_angle * weight_attribute * l2_distance)

    loss = tf.reduce_mean(l2_distance)

    return weighted_loss, loss


def L2Loss(label, landmark_pred):
    landmark_gt = tf.cast(label[:, :136], tf.float32)
    loss = tf.reduce_sum(tf.square(landmark_gt - landmark_pred), axis=1)

    return loss


def valid_loss(label, landmark_pred):
    landmark_gt = tf.cast(label[:, :136], tf.float32)
    loss = K.mean(K.sum((landmark_gt - landmark_pred) **2, 1))

    return loss


def WingLoss(label, landmark_pred, angle_pred, wing_w=10.0, wing_epsilon=2.0):
    landmark_gt, euler_angle_gt = tf.cast(label[:, :136], tf.float32), tf.cast(label[:, 142:], tf.float32)
    wing_c = wing_w * (1.0 - math.log(1.0 + wing_w / wing_epsilon))
    euler_angle_weights = 1 - tf.cos(euler_angle_gt - angle_pred)
    euler_angle_weights = K.sum(euler_angle_weights, 1)
    
    abs_error = K.abs(landmark_gt - landmark_pred)
    loss = tf.where(K.less(abs_error, wing_w), wing_w * K.log(1.0 + abs_error / wing_epsilon), abs_error - wing_c)
    loss_sum = K.sum(loss, 1)
    loss_sum *= euler_angle_weights

    return K.mean(loss_sum)