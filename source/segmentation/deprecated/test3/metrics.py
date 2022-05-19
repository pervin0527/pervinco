import tensorflow as tf

class Sparse_MeanIoU(tf.keras.metrics.MeanIoU):
  def __init__(self, y_true=None, y_pred=None, num_classes=None, name=None, dtype=None):
    super(Sparse_MeanIoU, self).__init__(num_classes = num_classes,name=name, dtype=dtype)

  def update_state(self, y_true, y_pred, sample_weight=None):
    y_pred = tf.math.argmax(y_pred, axis=-1)
    return super().update_state(y_true, y_pred, sample_weight)