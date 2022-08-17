import tensorflow as tf
from loss import centernet_loss
from keras_resnet import models as resnet_models
from tensorflow.keras import backend as K

def nms(heat, kernel=3):
    hmax = tf.nn.max_pool2d(heat, (kernel, kernel), strides=1, padding='SAME')
    heat = tf.where(tf.equal(hmax, heat), heat, tf.zeros_like(heat))
    return heat


def topk(hm, max_objects=100):
    hm = nms(hm)
    # (b, h * w * c)
    b, h, w, c = tf.shape(hm)[0], tf.shape(hm)[1], tf.shape(hm)[2], tf.shape(hm)[3]
    # hm2 = tf.transpose(hm, (0, 3, 1, 2))
    # hm2 = tf.reshape(hm2, (b, c, -1))
    hm = tf.reshape(hm, (b, -1))
    # (b, k), (b, k)
    scores, indices = tf.nn.top_k(hm, k=max_objects)
    # scores2, indices2 = tf.nn.top_k(hm2, k=max_objects)
    # scores2 = tf.reshape(scores2, (b, -1))
    # topk = tf.nn.top_k(scores2, k=max_objects)
    class_ids = indices % c
    xs = indices // c % w
    ys = indices // c // w
    indices = ys * w + xs
    return scores, indices, class_ids, xs, ys


def evaluate_batch_item(batch_item_detections, num_classes, max_objects_per_class=20, max_objects=100,
                        iou_threshold=0.5, score_threshold=0.1):
    batch_item_detections = tf.boolean_mask(batch_item_detections,
                                            tf.greater(batch_item_detections[:, 4], score_threshold))
    detections_per_class = []
    for cls_id in range(num_classes):
        class_detections = tf.boolean_mask(batch_item_detections, tf.equal(batch_item_detections[:, 5], cls_id))
        nms_keep_indices = tf.image.non_max_suppression(class_detections[:, :4],
                                                        class_detections[:, 4],
                                                        max_objects_per_class,
                                                        iou_threshold=iou_threshold)
        class_detections = K.gather(class_detections, nms_keep_indices)
        detections_per_class.append(class_detections)

    batch_item_detections = K.concatenate(detections_per_class, axis=0)

    def filter():
        nonlocal batch_item_detections
        _, indices = tf.nn.top_k(batch_item_detections[:, 4], k=max_objects)
        batch_item_detections_ = tf.gather(batch_item_detections, indices)
        return batch_item_detections_

    def pad():
        nonlocal batch_item_detections
        batch_item_num_detections = tf.shape(batch_item_detections)[0]
        batch_item_num_pad = tf.maximum(max_objects - batch_item_num_detections, 0)
        batch_item_detections_ = tf.pad(tensor=batch_item_detections,
                                        paddings=[
                                            [0, batch_item_num_pad],
                                            [0, 0]],
                                        mode='CONSTANT',
                                        constant_values=0.0)
        return batch_item_detections_

    batch_item_detections = tf.cond(tf.shape(batch_item_detections)[0] >= 100,
                                    filter,
                                    pad)
    return batch_item_detections


def decode(hm, wh, reg, max_objects=100, nms=True, flip_test=False, num_classes=20, score_threshold=0.1):
    if flip_test:
        hm = (hm[0:1] + hm[1:2, :, ::-1]) / 2
        wh = (wh[0:1] + wh[1:2, :, ::-1]) / 2
        reg = reg[0:1]
    scores, indices, class_ids, xs, ys = topk(hm, max_objects=max_objects)
    b = tf.shape(hm)[0]
    # (b, h * w, 2)
    reg = tf.reshape(reg, (b, -1, tf.shape(reg)[-1]))
    # (b, h * w, 2)
    wh = tf.reshape(wh, (b, -1, tf.shape(wh)[-1]))
    # (b, k, 2)
    topk_reg = tf.gather(reg, indices, batch_dims=1)
    # (b, k, 2)
    topk_wh = tf.cast(tf.gather(wh, indices, batch_dims=1), tf.float32)
    topk_cx = tf.cast(tf.expand_dims(xs, axis=-1), tf.float32) + topk_reg[..., 0:1]
    topk_cy = tf.cast(tf.expand_dims(ys, axis=-1), tf.float32) + topk_reg[..., 1:2]
    scores = tf.expand_dims(scores, axis=-1)
    class_ids = tf.cast(tf.expand_dims(class_ids, axis=-1), tf.float32)
    topk_x1 = topk_cx - topk_wh[..., 0:1] / 2
    topk_x2 = topk_cx + topk_wh[..., 0:1] / 2
    topk_y1 = topk_cy - topk_wh[..., 1:2] / 2
    topk_y2 = topk_cy + topk_wh[..., 1:2] / 2
    # (b, k, 6)
    detections = tf.concat([topk_x1, topk_y1, topk_x2, topk_y2, scores, class_ids], axis=-1)
    if nms:
        detections = tf.map_fn(lambda x: evaluate_batch_item(x[0],
                                                             num_classes=num_classes,
                                                             score_threshold=score_threshold),
                               elems=[detections],
                               dtype=tf.float32)
    return detections

def centernet(inputs, num_classes, max_detections, threshold, backbone_name="resnet18", freeze_bn=True):
    inputs = tf.keras.Input(shape=inputs)

    if backbone_name == "resnet18":
       backbone_model = resnet_models.ResNet18(inputs, include_top=False, freeze_bn=freeze_bn)
    elif backbone_name == 'resnet34':
        backbone_model = resnet_models.ResNet34(inputs, include_top=False, freeze_bn=freeze_bn)
    elif backbone_name == 'resnet50':
        backbone_model = resnet_models.ResNet50(inputs, include_top=False, freeze_bn=freeze_bn)
    elif backbone_name == 'resnet101':
        backbone_model = resnet_models.ResNet101(inputs, include_top=False, freeze_bn=freeze_bn)

    C5 = backbone_model.outputs[-1]
    x = tf.keras.layers.Dropout(rate=0.5)(C5)

    num_filters = 256
    for i in range(3):
        num_filters = num_filters // pow(2, i)

        x = tf.keras.layers.Conv2DTranspose(num_filters, (4, 4), strides=2, padding="same", kernel_initializer="he_normal", kernel_regularizer=tf.keras.regularizers.L2(5e-4), use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.nn.relu(x)

    y1 = tf.keras.layers.Conv2D(64, 3, padding="SAME", use_bias=False, kernel_initializer="he_normal", kernel_regularizer=tf.keras.regularizers.L2(5e-4))(x)
    y1 = tf.keras.layers.BatchNormalization()(y1)
    y1 = tf.nn.relu(y1)
    y1 = tf.keras.layers.Conv2D(num_classes, 1, kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.L2(5e-4), activation='sigmoid')(y1)

    y2 = tf.keras.layers.Conv2D(64, 3, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.L2(5e-4))(x)
    y2 = tf.keras.layers.BatchNormalization()(y2)
    y2 = tf.keras.layers.ReLU()(y2)
    y2 = tf.keras.layers.Conv2D(2, 1, kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.L2(5e-4))(y2)

    y3 = tf.keras.layers.Conv2D(64, 3, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.L2(5e-4))(x)
    y3 = tf.keras.layers.BatchNormalization()(y3)
    y3 = tf.keras.layers.ReLU()(y3)
    y3 = tf.keras.layers.Conv2D(2, 1, kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.L2(5e-4))(y3)

    model = tf.keras.Model(inputs=inputs, outputs=[y1, y2, y3])

    # detections = tf.keras.layers.Lambda(lambda x: decode(*x, max_objects=max_detections, score_threshold=threshold, nms=nms, num_classes=num_classes))([y1, y2, y3])
    # model = tf.keras.Model(inputs=inputs, outputs=[y1, y2, y3, detections])

    return model

class CenterNet(tf.keras.Model):
    def __init__(self, inputs, num_classes, max_detections, threshold, backbone_name="resnet18", freeze_bn=True):
        super(CenterNet, self).__init__()

        self.centernet = centernet(inputs, num_classes, max_detections, threshold, backbone_name, freeze_bn)
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.centernet.summary()

    def call(self, x, training=False):
        hm_pred, wh_pred, reg_pred = self.centernet(x)
        return hm_pred, wh_pred, reg_pred
        # hm_pred, wh_pred, reg_pred, detections = self.centernet(x)
        # return hm_pred, wh_pred, reg_pred, detections

    def train_step(self, data):
        image, hm, wh, reg, reg_mask, indices = data["image"], data["hm"], data["wh"], data["reg"], data["reg_mask"], data["indices"]

        with tf.GradientTape() as tape:
            hm_pred, wh_pred, reg_pred = self(image, training=True)
            # hm_pred, wh_pred, reg_pred, detections = self(image, training=True)
            loss = centernet_loss(hm_pred, wh_pred, reg_pred, hm, wh, reg, reg_mask, indices)
        
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.loss_tracker.update_state(loss)

        return {"loss" : self.loss_tracker.result()}

    def test_step(self, data):
        image, hm, wh, reg, reg_mask, indices = data["image"], data["hm"], data["wh"], data["reg"], data["reg_mask"], data["indices"]
        hm_pred, wh_pred, reg_pred = self(image, training=False)
        # hm_pred, wh_pred, reg_pred, detections = self(image, training=False)
        loss = centernet_loss(hm_pred, wh_pred, reg_pred, hm, wh, reg, reg_mask, indices)
        self.loss_tracker.update_state(loss)

        return {"loss" : self.loss_tracker.result()}