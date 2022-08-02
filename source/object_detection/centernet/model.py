import numpy as np
import tensorflow as tf
from loss import compute_loss
from backbone import resnet18, resnet101

def conv_bn_act(inputs, filters, kernel_size, strides=1, padding="same", activation=tf.nn.relu, use_bn=True):
    if use_bn:
        conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=False)(inputs)
        conv = tf.keras.layers.BatchNormalization()(conv)
    else:
        conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=True)(inputs)

    if activation != None:
        conv = activation(conv)

    return conv


def upsampling(inputs,  method="deconv"):
    assert method in ["resize", "deconv"]

    if method == "resize":
        input_shape = tf.shape(inputs)
        output = tf.image.resize(inputs, (input_shape[1] * 2, input_shape[2] * 2))

    if method == "deconv":
        numm_filter = inputs.shape.as_list()[-1]
        output = tf.layers.Conv2DTranspose(inputs=inputs, filters=numm_filter, kernel_size=4, strides=2, padding='same')
    
    return output


def centernet(input_shape, num_classes, backbone='resnet18'):
    input_layer = tf.keras.Input(shape=input_shape)
    
    if backbone == "resnet18":
        c2, c3, c4, c5 = resnet18()(input_layer)
    elif backbone == "resnet101":
        c2, c3, c4, c5 = resnet101()(input_layer)

    p5 = conv_bn_act(c5, filters=128, kernel_size=[1, 1])

    up_p5 = upsampling(p5, method='resize')
    reduce_dim_c4 = conv_bn_act(c4, filters=128, kernel_size=[1, 1])
    p4 = 0.5 * up_p5 + 0.5 * reduce_dim_c4

    up_p4 = upsampling(p4, method='resize')
    reduce_dim_c3 = conv_bn_act(c3, filters=128, kernel_size=[1, 1])
    p3 = 0.5 * up_p4 + 0.5 * reduce_dim_c3

    up_p3 = upsampling(p3, method='resize')
    reduce_dim_c2 = conv_bn_act(c2, filters=128, kernel_size=[1, 1])
    p2 = 0.5 * up_p3 + 0.5 * reduce_dim_c2

    features = conv_bn_act(p2, filters=128, kernel_size=[3, 3])

    y1 = conv_bn_act(features, filters=64, kernel_size=[3, 3])
    y1 = tf.keras.layers.Conv2D(num_classes, 1, 1, padding='valid', activation = tf.nn.sigmoid, bias_initializer=tf.constant_initializer(-np.log(99.)), name='y1')(y1)

    y2 = conv_bn_act(features, filters=64, kernel_size=[3, 3])
    y2 = tf.keras.layers.Conv2D(2, 1, 1, padding='valid', activation = None, name='y2')(y2)
    
    y3 =  conv_bn_act(features, filters=64, kernel_size=[3, 3])
    y3 = tf.keras.layers.Conv2D(2, 1, 1, padding='valid', activation = None, name='y3')(y3)

    model = tf.keras.models.Model(inputs=input_layer, outputs=[y1, y2, y3])

    return model


class CenterNet(tf.keras.Model):
    def __init__(self, inputs, num_classes, backbone):
        super(CenterNet, self).__init__()
        self.inputs = inputs
        self.num_classes = num_classes
        self.backbone = backbone

        self.model = centernet(input_shape=self.inputs, num_classes=self.num_classes, backbone=self.backbone)
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    def call(self, x, training=False):
        hm_pred, wh_pred, reg_pred = self.model(x)

        return hm_pred, wh_pred, reg_pred

    def train_step(self, data):
        image, hm_gt, wh_gt, reg_gt, reg_mask, ind = data[0], data[1][0], data[1][1], data[1][2], data[1][3], data[1][4]

        with tf.GradientTape() as tape:
            hm_pred, wh_pred, reg_pred = self(image, training=True)
            loss = compute_loss(hm_pred, wh_pred, reg_pred, hm_gt, wh_gt, reg_gt, reg_mask, ind)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.loss_tracker.update_state(loss)
        
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        image, hm_gt, wh_gt, reg_gt, reg_mask, ind = data[0], data[1][0], data[1][1], data[1][2], data[1][3], data[1][4]
        hm_pred, wh_pred, reg_pred = self(image)
        loss = compute_loss(hm_pred, wh_pred, reg_pred, hm_gt, wh_gt, reg_gt, reg_mask, ind)

        self.loss_tracker.update_state(loss)
        return {"loss" : self.loss_tracker.result()}

    @property
    def metrics(self):
        return [self.loss_tracker]


def nms(heat, kernel=3):
    hmax = tf.keras.layers.MaxPooling2D((kernel, kernel), strides=1, padding='same')(heat)
    heat = tf.where(tf.equal(hmax, heat), heat, tf.zeros_like(heat))

    return heat


def topk(hm, max_detections):
    hm = nms(hm)
    b, h, w, c = tf.shape(hm)[0], tf.shape(hm)[1], tf.shape(hm)[2], tf.shape(hm)[3]
    hm = tf.reshape(hm, (b, -1))
    scores, indices = tf.math.top_k(hm, k=max_detections, sorted=True)

    class_ids = indices % c
    xs = indices // c % w
    ys = indices // c // w
    indices = ys * w + xs
    
    return scores, indices, class_ids, xs, ys


def decode(hm, wh, reg, max_detections):
    scores, indices, class_ids, xs, ys = topk(hm, max_detections=max_detections)
    b = tf.shape(hm)[0]
    
    reg = tf.reshape(reg, [b, -1, 2])
    wh = tf.reshape(wh, [b, -1, 2])
    length = tf.shape(wh)[1]

    batch_idx = tf.expand_dims(tf.range(0, b), 1)
    batch_idx = tf.tile(batch_idx, (1, max_detections))
    full_indices = tf.reshape(batch_idx, [-1]) * tf.cast(length, tf.int32) + tf.reshape(indices, [-1])
                    
    topk_reg = tf.gather(tf.reshape(reg, [-1,2]), full_indices)
    topk_reg = tf.reshape(topk_reg, [b, -1, 2])
    
    topk_wh = tf.gather(tf.reshape(wh, [-1,2]), full_indices)
    topk_wh = tf.reshape(topk_wh, [b, -1, 2])

    topk_cx = tf.cast(tf.expand_dims(xs, axis=-1), tf.float32) + topk_reg[..., 0:1]
    topk_cy = tf.cast(tf.expand_dims(ys, axis=-1), tf.float32) + topk_reg[..., 1:2]

    topk_x1, topk_y1 = topk_cx - topk_wh[..., 0:1] / 2, topk_cy - topk_wh[..., 1:2] / 2
    topk_x2, topk_y2 = topk_cx + topk_wh[..., 0:1] / 2, topk_cy + topk_wh[..., 1:2] / 2
    
    scores = tf.expand_dims(scores, axis=-1)
    class_ids = tf.cast(tf.expand_dims(class_ids, axis=-1), tf.float32)

    detections = tf.concat([topk_x1, topk_y1, topk_x2, topk_y2, scores, class_ids], axis=-1)

    return detections