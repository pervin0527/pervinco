import tensorflow as tf

def hard_negative_mining(loss, gt_confs, neg_ratio):
    pos_idx = gt_confs > 0
    num_pos = tf.reduce_sum(tf.dtypes.cast(pos_idx, tf.int32), axis=1)
    num_neg = num_pos * neg_ratio

    rank = tf.argsort(loss, axis=1, direction="DESCENDING")
    rank = tf.argsort(rank, axis=1)
    neg_idx = rank < tf.expand_dims(num_neg, 1)

    return pos_idx, neg_idx

class SSDLosses(object):
    def __init__(self, neg_ratio, num_classes):
        self.neg_ratio = neg_ratio
        self.num_classes = num_classes

    def __call__(self, confs, locs, gt_confs, gt_locs):
        cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        temp_loss = cross_entropy(gt_confs, confs)
        pos_idx, neg_idx = hard_negative_mining(temp_loss, gt_confs, self.neg_ratio)

        cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='sum')
        smooth_l1_loss = tf.keras.losses.Huber(reduction='sum')

        conf_loss = cross_entropy(gt_confs[tf.math.logical_or(pos_idx, neg_idx)], confs[tf.math.logical_or(pos_idx, neg_idx)])
        loc_loss = smooth_l1_loss(gt_locs[pos_idx], locs[pos_idx])

        num_pos = tf.reduce_sum(tf.dtypes.cast(pos_idx, tf.float32))

        conf_loss = conf_loss / num_pos
        loc_loss = loc_loss / num_pos

        return conf_loss, loc_loss