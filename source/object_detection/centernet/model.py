import numpy as np
import tensorflow as tf
from loss import compute_loss
# from backbone import resnet18, resnet101
from keras_resnet import models as resnet_models

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
    

def CenterNet(input_shape, num_classes, max_detections, backbone='resnet18'):
    output_size = input_shape[0] // 4
    image_input = tf.keras.Input(shape=(None, None, 3))
    hm_input = tf.keras.Input(shape=(output_size, output_size, num_classes))
    wh_input = tf.keras.Input(shape=(max_detections, 2))
    reg_input = tf.keras.Input(shape=(max_detections, 2))
    reg_mask_input = tf.keras.Input(shape=(max_detections,))
    index_input = tf.keras.Input(shape=(max_detections,))
    
    if backbone == "resnet18":
        resnet = resnet_models.ResNet18(image_input, include_top=False, freeze_bn=False)

    elif backbone == "resnet101":
        resnet = resnet_models.ResNet101(image_input, include_top=False, freeze_bn=False)

    C5 = resnet.output[-1]
    x = tf.keras.layers.Dropout(rate=0.5)(C5)
    
    # decoder
    num_filters = 256
    for i in range(3):
        num_filters = num_filters // pow(2, i)
        x = tf.keras.layers.Conv2DTranspose(num_filters, (4, 4), strides=2, use_bias=False, padding='same',
                                            kernel_initializer='he_normal',
                                            kernel_regularizer=tf.keras.regularizers.L2(5e-4))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

    # hm header
    y1 = tf.keras.layers.Conv2D(64, 3, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.L2(5e-4))(x)
    y1 = tf.keras.layers.BatchNormalization()(y1)
    y1 = tf.keras.layers.ReLU()(y1)
    y1 = tf.keras.layers.Conv2D(num_classes, 1, kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.L2(5e-4), activation='sigmoid')(y1)

    # wh header
    y2 = tf.keras.layers.Conv2D(64, 3, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.L2(5e-4))(x)
    y2 = tf.keras.layers.BatchNormalization()(y2)
    y2 = tf.keras.layers.ReLU()(y2)
    y2 = tf.keras.layers.Conv2D(2, 1, kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.L2(5e-4))(y2)

    # reg header
    y3 = tf.keras.layers.Conv2D(64, 3, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.L2(5e-4))(x)
    y3 = tf.keras.layers.BatchNormalization()(y3)
    y3 = tf.keras.layers.ReLU()(y3)
    y3 = tf.keras.layers.Conv2D(2, 1, kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.L2(5e-4))(y3)

    loss_ = tf.keras.layers.Lambda(compute_loss, name='centernet_loss')([y1, y2, y3, hm_input, wh_input, reg_input, reg_mask_input, index_input])
    model = tf.keras.Model(inputs=[image_input, hm_input, wh_input, reg_input, reg_mask_input, index_input], outputs=[loss_])

    # detections = decode(y1, y2, y3)
    detections = tf.keras.layers.Lambda(lambda x: decode(*x, max_detections=max_detections))([y1, y2, y3])
    prediction_model = tf.keras.Model(inputs=image_input, outputs=detections)

    return model, prediction_model