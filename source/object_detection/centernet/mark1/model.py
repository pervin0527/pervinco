import tensorflow as tf
from keras_resnet import models as resnet_models
from losses import total_loss
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import RandomNormal, Constant
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2DTranspose, BatchNormalization, ReLU, Conv2D, Lambda, MaxPooling2D, Dropout, ZeroPadding2D

def nms(heat, kernel=3):
    hmax = MaxPooling2D((kernel, kernel), strides=1, padding='same')(heat)
    heat = tf.where(tf.equal(hmax, heat), heat, tf.zeros_like(heat))
    
    return heat

def topk(hm, max_objects=100):
    hm = nms(hm)
    b, h, w, c = tf.shape(hm)[0], tf.shape(hm)[1], tf.shape(hm)[2], tf.shape(hm)[3]
    hm = tf.reshape(hm, (b, -1))
    scores, indices = tf.nn.top_k(hm, k=max_objects)
    class_ids = indices % c
    xs = indices // c % w
    ys = indices // c // w
    indices = ys * w + xs

    return scores, indices, class_ids, xs, ys

def decode(hm, wh, reg, max_objects=100):
    scores, indices, class_ids, xs, ys = topk(hm, max_objects=max_objects)
    b = tf.shape(hm)[0]
    reg = tf.reshape(reg, (b, -1, tf.shape(reg)[-1]))
    wh = tf.reshape(wh, (b, -1, tf.shape(wh)[-1]))
    topk_reg = tf.gather(reg, indices, batch_dims=1)

    scores = tf.expand_dims(scores, axis=-1)
    class_ids = tf.cast(tf.expand_dims(class_ids, axis=-1), tf.float32)
    
    topk_wh = tf.cast(tf.gather(wh, indices, batch_dims=1), tf.float32)
    topk_cx = tf.cast(tf.expand_dims(xs, axis=-1), tf.float32) + topk_reg[..., 0:1]
    topk_cy = tf.cast(tf.expand_dims(ys, axis=-1), tf.float32) + topk_reg[..., 1:2]
    
    topk_x1 = topk_cx - topk_wh[..., 0:1] / 2
    topk_x2 = topk_cx + topk_wh[..., 0:1] / 2
    topk_y1 = topk_cy - topk_wh[..., 1:2] / 2
    topk_y2 = topk_cy + topk_wh[..., 1:2] / 2
    
    detections = tf.concat([topk_x1, topk_y1, topk_x2, topk_y2, scores, class_ids], axis=-1)

    return detections

def centernet(num_classes, backbone="resnet101", input_size=512, max_detections=10, freeze_bn=True):
    output_size = input_size // 4
    image_input = Input(shape=(None, None, 3))
    hm_input = Input(shape=(output_size, output_size, num_classes))
    wh_input = Input(shape=(max_detections, 2))
    reg_input = Input(shape=(max_detections, 2))
    reg_mask_input = Input(shape=(max_detections,))
    index_input = Input(shape=(max_detections,))

    resnet = resnet_models.ResNet101(image_input, include_top=False, freeze_bn=freeze_bn)
    C5 = resnet.outputs[-1]
    x = Dropout(rate=0.5)(C5)

    # C5 = tf.keras.applications.ResNet101(input_tensor=image_input, include_top=False)
    # x = Dropout(rate=0.5)(C5.output)
    
    num_filters = 256
    for i in range(3):
        num_filters = num_filters // pow(2, i)
        x = Conv2DTranspose(num_filters, (4, 4), strides=2, use_bias=False, padding='same',
                            kernel_initializer='he_normal',
                            kernel_regularizer=l2(5e-4))(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

    # hm header
    y1 = Conv2D(64, 3, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(x)
    y1 = BatchNormalization()(y1)
    y1 = ReLU()(y1)
    y1 = Conv2D(num_classes, 1, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4), activation='sigmoid')(y1)

    # wh header
    y2 = Conv2D(64, 3, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(x)
    y2 = BatchNormalization()(y2)
    y2 = ReLU()(y2)
    y2 = Conv2D(2, 1, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(y2)

    # reg header
    y3 = Conv2D(64, 3, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(x)
    y3 = BatchNormalization()(y3)
    y3 = ReLU()(y3)
    y3 = Conv2D(2, 1, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(y3)

    loss_ = Lambda(total_loss, name='centernet_loss')([y1, y2, y3, hm_input, wh_input, reg_input, reg_mask_input, index_input])
    model = Model(inputs=[image_input, hm_input, wh_input, reg_input, reg_mask_input, index_input], outputs=[loss_])

    # detections = decode(y1, y2, y3)
    detections = Lambda(lambda x: decode(*x, max_objects=max_detections))([y1, y2, y3])
    prediction_model = Model(inputs=image_input, outputs=detections)

    return model, prediction_model

if __name__ == "__main__":
    model, prediction_model = centernet(1, "resnet101", 512, 10)
    model.summary()