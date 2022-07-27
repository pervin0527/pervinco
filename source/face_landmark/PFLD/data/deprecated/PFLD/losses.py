import math
import tensorflow as tf
from tensorflow.keras import backend as K


def PFLDLoss():
    def _PFLDLoss(y_true, y_pred):
        batch_size = tf.cast(K.shape(y_pred)[0], tf.float32)
        landmark_pred, angle_pred = y_pred[:, :136], y_pred[:, 136:]
        landmark_gt, attribute_gt, euler_angle_gt = tf.cast(y_true[:, :136], tf.float32), tf.cast(y_true[:, 136:142], tf.float32), tf.cast(y_true[:, 142:], tf.float32)
        
        weight_angle = K.sum(1 - tf.cos(angle_pred - euler_angle_gt), axis=1)
        attributes_w_n = tf.cast(attribute_gt[:, 1:6], tf.float32)

        mat_ratio = K.mean(attributes_w_n, axis=0)
        N = K.shape(mat_ratio)[0]
        mat_ratio = tf.where(mat_ratio>0, 1.0/mat_ratio, batch_size)
        weight_attribute = K.sum(tf.matmul(attributes_w_n, K.reshape(mat_ratio, (N,1))), axis=1)
        
        l2_distant = K.sum((landmark_gt - landmark_pred) * (landmark_gt - landmark_pred), axis=1)
        loss = tf.reduce_mean(l2_distant)
        weighted_loss = tf.reduce_mean(weight_angle * weight_attribute * l2_distant)

        return weighted_loss

    return _PFLDLoss