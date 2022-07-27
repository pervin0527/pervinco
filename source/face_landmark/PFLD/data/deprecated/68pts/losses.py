import math
import tensorflow as tf
import tensorflow.keras.backend as K


def PFLDLoss():
    def _PFLDLoss(y_true, y_pred):
        train_batchsize = tf.cast(K.shape(y_pred)[0], tf.float32)
        landmarks, angle = y_pred[:, :136], y_pred[:, 136:]

        landmark_gt, attribute_gt, euler_angle_gt = tf.cast(y_true[:,:136], tf.float32),tf.cast(y_true[:,136:142], tf.float32),tf.cast(y_true[:,142:],tf.float32)
        weight_angle = K.sum(1 - tf.cos(angle - euler_angle_gt), axis=1) # [8,]

        attributes_w_n = tf.cast(attribute_gt[:, 1:6], tf.float32)
        mat_ratio = K.mean(attributes_w_n, axis=0)
        N = K.shape(mat_ratio)[0]
        mat_ratio = tf.where(mat_ratio>0, 1.0/mat_ratio, train_batchsize)
        weight_attribute = K.sum(tf.matmul(attributes_w_n, K.reshape(mat_ratio, (N,1))), axis=1) # [8,1]
        l2_distant = K.sum((landmark_gt - landmarks) * (landmark_gt - landmarks), axis=1)
        
        return K.mean(weight_angle * weight_attribute * l2_distant)
    
    return _PFLDLoss


def L2Loss():
    def _L2Loss(y_true, y_pred):
        landmarks, angle = y_pred[:, :136], y_pred[:, 136:]
        landmark_gt, _, _ = tf.cast(y_true[:,:136], tf.float32),tf.cast(y_true[:,136:142], tf.float32),tf.cast(y_true[:,142:],tf.float32)
        l2_distant = tf.reduce_sum((landmark_gt - landmarks) * (landmark_gt - landmarks), axis=1)
        
        return tf.reduce_mean(l2_distant)

    return _L2Loss


def WingLoss(w=10.0, epsilon=2.0):
    def _WingLoss(y_true, y_pred):
        landmarks, angle = y_pred[:, :136], y_pred[:, 136:]
        landmark_gt, attribute_gt, euler_angle_gt = tf.cast(y_true[:,:136], tf.float32),tf.cast(y_true[:,136:142], tf.float32),tf.cast(y_true[:,142:],tf.float32)

        c = w * (1.0 - math.log(1.0 + w / epsilon))
        weight_angle = K.sum(1 - tf.cos(angle - euler_angle_gt), axis=1) # [8,]

        abs_error = tf.abs(landmark_gt - landmarks)
        loss = tf.where(tf.less(w, abs_error), w * tf.math.log(1.0 + abs_error / epsilon), abs_error - c)
        loss_sum = K.sum(loss, 1)
        loss_sum *= weight_angle
        
        return tf.reduce_mean(loss_sum)
    
    return _WingLoss