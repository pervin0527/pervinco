import tensorflow as tf
from models.losses import loss
from models.hourglass import HourglassNetwork
from models.resnet import ResNet50, centernet_head


def nms(heat, kernel=3):
    hmax = tf.keras.layers.MaxPooling2D((kernel, kernel), strides=1, padding='SAME')(heat)
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


def decode(hm, wh, reg, max_objects=100):
    scores, indices, class_ids, xs, ys = topk(hm, max_objects=max_objects)
    b = tf.shape(hm)[0]

    reg = tf.reshape(reg, [b, -1, 2])
    wh = tf.reshape(wh, [b, -1, 2])
    length = tf.shape(wh)[1]

    batch_idx = tf.expand_dims(tf.range(0, b), 1)
    batch_idx = tf.tile(batch_idx, (1, max_objects))
    full_indices = tf.reshape(batch_idx, [-1]) * tf.cast(length, dtype=tf.int32) + tf.reshape(indices, [-1])
                    
    topk_reg = tf.gather(tf.reshape(reg, [-1, 2]), full_indices)
    topk_reg = tf.reshape(topk_reg, [b, -1, 2])
    
    topk_wh = tf.gather(tf.reshape(wh, [-1, 2]), full_indices)
    topk_wh = tf.reshape(topk_wh, [b, -1, 2])

    topk_cx = tf.cast(tf.expand_dims(xs, axis=-1), tf.float32) + topk_reg[..., 0:1]
    topk_cy = tf.cast(tf.expand_dims(ys, axis=-1), tf.float32) + topk_reg[..., 1:2]

    topk_x1, topk_y1 = topk_cx - topk_wh[..., 0:1] / 2, topk_cy - topk_wh[..., 1:2] / 2
    topk_x2, topk_y2 = topk_cx + topk_wh[..., 0:1] / 2, topk_cy + topk_wh[..., 1:2] / 2
    
    scores = tf.expand_dims(scores, axis=-1)
    class_ids = tf.cast(tf.expand_dims(class_ids, axis=-1), tf.float32)
    
    detections = tf.concat([topk_x1, topk_y1, topk_x2, topk_y2, scores, class_ids], axis=-1)

    return detections


def centernet(input_shape, num_classes, backbone='resnet50', max_objects=100, mode="train", num_stacks=2):
    assert backbone in ['resnet50', 'hourglass']
    # image_input = tf.keras.Input(shape=input_shape)
    # preprocess_input = tf.keras.layers.Lambda(lambda x : tf.cast(x, tf.float32) / 127.5 - 1)(image_input)
    image_input = tf.keras.Input(shape=input_shape)

    if backbone=='resnet50':
        # C5 = ResNet50(image_input)
        resnet = tf.keras.applications.resnet_v2.ResNet50V2(include_top=False, weights="imagenet", input_tensor=image_input, classes=num_classes, pooling=None, classifier_activation=None)
        C5 = resnet.output
        y1, y2, y3 = centernet_head(C5, num_classes)

        if mode=="train":
            model = tf.keras.Model(inputs=image_input, outputs=[y1, y2, y3])

            detections = tf.keras.layers.Lambda(lambda x: decode(*x, max_objects=max_objects))([y1, y2, y3])
            prediction_model = tf.keras.Model(inputs=image_input, outputs=detections)
            return model, prediction_model

        elif mode=="predict":
            detections = tf.keras.layers.Lambda(lambda x: decode(*x, max_objects=max_objects))([y1, y2, y3])
            prediction_model = tf.keras.Model(inputs=image_input, outputs=detections)
            return prediction_model

        elif mode=="heatmap":
            prediction_model = tf.keras.Model(inputs=image_input, outputs=y1)
            return prediction_model

    else:
        outs = HourglassNetwork(image_input,num_stacks,num_classes)
        if mode=="train":
            temp_outs = []
            for out in outs:
                temp_outs += out
            model = tf.keras.Model(inputs=image_input, outputs=temp_outs)
            y1, y2, y3 = outs[-1]
            detections = tf.keras.layers.Lambda(lambda x: decode(*x, max_objects=max_objects))([y1, y2, y3])
            prediction_model = tf.keras.Model(inputs=image_input, outputs=[detections])
            return model, prediction_model
        
        elif mode=="predict":
            y1, y2, y3 = outs[-1]
            detections = tf.keras.layers.Lambda(lambda x: decode(*x, max_objects=max_objects))([y1, y2, y3])
            prediction_model = tf.keras.Model(inputs=image_input, outputs=[detections])
            return prediction_model

        elif mode=="heatmap":
            y1, y2, y3 = outs[-1]
            prediction_model = tf.keras.Model(inputs=image_input, outputs=y1)
            return prediction_model


def get_train_model(model_body, input_shape, num_classes, backbone='resnet50', max_objects=100):
    output_size = input_shape[0] // 4
    hm_input = tf.keras.Input(shape=(output_size, output_size, num_classes))
    wh_input = tf.keras.Input(shape=(max_objects, 2))
    reg_input = tf.keras.Input(shape=(max_objects, 2))
    reg_mask_input = tf.keras.Input(shape=(max_objects,))
    index_input = tf.keras.Input(shape=(max_objects,))

    if backbone=='resnet50':
        y1, y2, y3 = model_body.output
        loss_ = tf.keras.layers.Lambda(loss, output_shape = (1, ),name='centernet_loss')([y1, y2, y3, hm_input, wh_input, reg_input, reg_mask_input, index_input])
        model = tf.keras.Model(inputs=[model_body.input, hm_input, wh_input, reg_input, reg_mask_input, index_input], outputs=[loss_])
    
    else:
        outs = model_body.output
        loss_all = []
        for i in range(len(outs) // 3):  
            y1, y2, y3 = outs[0 + i * 3], outs[1 + i * 3], outs[2 + i * 3]
            loss_ = tf.keras.layers.Lambda(loss)([y1, y2, y3, hm_input, wh_input, reg_input, reg_mask_input, index_input])
            loss_all.append(loss_)
        loss_all = tf.keras.layers.Lambda(tf.reduce_mean, output_shape = (1, ),name='centernet_loss')(loss_)

        model = tf.keras.Model(inputs=[model_body.input, hm_input, wh_input, reg_input, reg_mask_input, index_input], outputs=[loss_all])
        
    return model
