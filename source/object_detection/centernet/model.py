import tensorflow as tf
from losses import total_loss
from backbone import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import L2
from tensorflow.keras.initializers import RandomNormal, Constant
from tensorflow.keras.layers import Input, MaxPooling2D, Lambda, Dropout, BatchNormalization, Conv2DTranspose, Conv2D, Activation

def nms(heat, kernel=3):
    hmax = MaxPooling2D((kernel, kernel), strides=1, padding='same')(heat)
    heat = tf.where(tf.equal(hmax, heat), heat, tf.zeros_like(heat))

    return heat


def topk(hm, max_objects=100):
    hm = nms(hm)
    b, h, w, c = tf.shape(hm)[0], tf.shape(hm)[1], tf.shape(hm)[2], tf.shape(hm)[3]
    hm = tf.reshape(hm, (b, -1))
    scores, indices = tf.math.top_k(hm, k=max_objects, sorted=True)

    class_ids = indices % c
    xs = indices // c % w
    ys = indices // c // w
    indices = ys * w + xs
    
    return scores, indices, class_ids, xs, ys


def decode(hm, wh, reg, max_objects=100, num_classes=20):
    scores, indices, class_ids, xs, ys = topk(hm, max_objects=max_objects)
    b = tf.shape(hm)[0]
    
    reg = tf.reshape(reg, [b, -1, 2])
    wh = tf.reshape(wh, [b, -1, 2])
    length = tf.shape(wh)[1]

    batch_idx = tf.expand_dims(tf.range(0, b), 1)
    batch_idx = tf.tile(batch_idx, (1, max_objects))
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


def centernet_head(x,num_classes):
    x = Dropout(rate=0.5)(x)
    num_filters = 256

    # 16, 16, 2048  ->  32, 32, 256 -> 64, 64, 128 -> 128, 128, 64
    for i in range(3):
        x = Conv2DTranspose(num_filters // pow(2, i), (4, 4), strides=2, use_bias=False, padding='same', kernel_initializer='he_normal', kernel_regularizer=L2(5e-4))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

    # hm header
    y1 = Conv2D(64, 3, padding='same', use_bias=False, kernel_initializer=RandomNormal(stddev=0.02))(x)
    y1 = BatchNormalization()(y1)
    y1 = Activation('relu')(y1)
    y1 = Conv2D(num_classes, 1, kernel_initializer=Constant(0), bias_initializer=Constant(-2.19), activation='sigmoid')(y1)

    # wh header
    y2 = Conv2D(64, 3, padding='same', use_bias=False, kernel_initializer=RandomNormal(stddev=0.02))(x)
    y2 = BatchNormalization()(y2)
    y2 = Activation('relu')(y2)
    y2 = Conv2D(2, 1, kernel_initializer=RandomNormal(stddev=0.02))(y2)

    # reg header
    y3 = Conv2D(64, 3, padding='same', use_bias=False, kernel_initializer=RandomNormal(stddev=0.02))(x)
    y3 = BatchNormalization()(y3)
    y3 = Activation('relu')(y3)
    y3 = Conv2D(2, 1, kernel_initializer=RandomNormal(stddev=0.02))(y3)

    return y1, y2, y3


def centernet(input_shape, num_classes, backbone='resnet50', max_objects=100, mode="train", num_stacks=2):
    assert backbone in ['resnet50', 'hourglass']

    output_size = input_shape[0] // 4
    image_input = Input(shape=input_shape)
    hm_input = Input(shape=(output_size, output_size, num_classes))
    wh_input = Input(shape=(max_objects, 2))
    reg_input = Input(shape=(max_objects, 2))
    reg_mask_input  = Input(shape=(max_objects,))
    index_input = Input(shape=(max_objects,))

    if backbone=='resnet50':
        C5 = ResNet50(image_input)
        y1, y2, y3 = centernet_head(C5, num_classes)

        if mode=="train":
            loss_ = Lambda(total_loss, name='centernet_loss')([y1, y2, y3, hm_input, wh_input, reg_input, reg_mask_input, index_input])
            model = Model(inputs=[image_input, hm_input, wh_input, reg_input, reg_mask_input, index_input], outputs=[loss_])
            
            detections = Lambda(lambda x: decode(*x, max_objects=max_objects))([y1, y2, y3])
            prediction_model = Model(inputs=image_input, outputs=detections)
        
            return model, prediction_model
        
        elif mode=="predict":
            detections = Lambda(lambda x: decode(*x, num_classes=num_classes, max_objects=max_objects))([y1, y2, y3])
            prediction_model = Model(inputs=image_input, outputs=detections)
        
            return prediction_model
        
        elif mode=="heatmap":
            prediction_model = Model(inputs=image_input, outputs=y1)
            
            return prediction_model