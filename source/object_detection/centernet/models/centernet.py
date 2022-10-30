import tensorflow as tf
from models.losses import loss
from models.resnet import centernet_head


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
    
    # scores = tf.expand_dims(scores, axis=-1)
    # class_ids = tf.cast(tf.expand_dims(class_ids, axis=-1), tf.float32)
    
    # detections = tf.concat([topk_x1, topk_y1, topk_x2, topk_y2, scores, class_ids], axis=-1)
    # return detections

    boundig_boxes = tf.concat([topk_x1, topk_y1, topk_x2, topk_y2], axis=-1)
    class_ids = tf.cast(class_ids, tf.float32)
    scores = tf.cast(scores, tf.float32)
    valid_detection = tf.expand_dims((max_objects), axis=0)
    
    return boundig_boxes, class_ids, scores, valid_detection


def centernet(input_shape, num_classes, backbone='resnet50', max_objects=100, weights="imagenet", mode="train"):
    assert backbone in ['resnet50', 'resnet101', 'mobilenet']
    print(weights)

    image_input = tf.keras.Input(shape=input_shape)

    if backbone=='resnet50':
        preprocess_input = tf.keras.layers.Lambda(lambda x : tf.keras.applications.resnet50.preprocess_input(x))(image_input)
        resnet = tf.keras.applications.resnet50.ResNet50(include_top=False,
                                                         weights=weights,
                                                         input_tensor=preprocess_input, 
                                                         classes=num_classes, 
                                                         pooling=None, 
                                                         classifier_activation=None)
        C5 = resnet.output
        y1, y2, y3 = centernet_head(C5, num_classes)

        if mode=="train":
            model = tf.keras.Model(inputs=image_input, outputs=[y1, y2, y3])

            # detections = tf.keras.layers.Lambda(lambda x: decode(*x, max_objects=max_objects))([y1, y2, y3])
            # prediction_model = tf.keras.Model(inputs=image_input, outputs=detections)
            # return model, prediction_model

            bboxes, classes, scores, valid_detection = tf.keras.layers.Lambda(lambda x: decode(*x, max_objects=max_objects))([y1, y2, y3])
            prediction_model = tf.keras.Model(inputs=image_input, outputs=[bboxes, classes, scores, valid_detection])

            return model, prediction_model

        elif mode=="predict":
            # detections = tf.keras.layers.Lambda(lambda x: decode(*x, max_objects=max_objects))([y1, y2, y3])
            # prediction_model = tf.keras.Model(inputs=image_input, outputs=detections)
            # return prediction_model

            bboxes, classes, scores, valid_detection = tf.keras.layers.Lambda(lambda x: decode(*x, max_objects=max_objects))([y1, y2, y3])
            prediction_model = tf.keras.Model(inputs=image_input, outputs=[bboxes, classes, scores, valid_detection])

            return prediction_model


    elif backbone=='resnet101':
        preprocess_input = tf.keras.layers.Lambda(lambda x : tf.keras.applications.resnet50.preprocess_input(x))(image_input)
        resnet = tf.keras.applications.resnet.ResNet101(include_top=False,
                                                        weights=weights,
                                                        input_tensor=preprocess_input, 
                                                        classes=num_classes, 
                                                        pooling=None, 
                                                        classifier_activation=None)
        C5 = resnet.output
        y1, y2, y3 = centernet_head(C5, num_classes)

        if mode=="train":
            model = tf.keras.Model(inputs=image_input, outputs=[y1, y2, y3])

            bboxes, classes, scores, valid_detection = tf.keras.layers.Lambda(lambda x: decode(*x, max_objects=max_objects))([y1, y2, y3])
            prediction_model = tf.keras.Model(inputs=image_input, outputs=[bboxes, classes, scores, valid_detection])

            return model, prediction_model

        elif mode=="predict":
            bboxes, classes, scores, valid_detection = tf.keras.layers.Lambda(lambda x: decode(*x, max_objects=max_objects))([y1, y2, y3])
            prediction_model = tf.keras.Model(inputs=image_input, outputs=[bboxes, classes, scores, valid_detection])

            return prediction_model


    elif backbone == "mobilenet":
        preprocess_input = tf.keras.layers.Lambda(lambda x : tf.keras.applications.mobilenet_v2.preprocess_input(x))(image_input)
        mobilenet = tf.keras.applications.mobilenet_v2.MobileNetV2(include_top=False,
                                                                   weights=weights,
                                                                   input_tensor=preprocess_input,
                                                                   classes=num_classes,
                                                                   pooling=None,
                                                                   classifier_activation=None)
        C5 = mobilenet.output
        y1, y2, y3 = centernet_head(C5, num_classes)

        if mode=="train":
            model = tf.keras.Model(inputs=image_input, outputs=[y1, y2, y3])

            bboxes, classes, scores, valid_detection = tf.keras.layers.Lambda(lambda x: decode(*x, max_objects=max_objects))([y1, y2, y3])
            prediction_model = tf.keras.Model(inputs=image_input, outputs=[bboxes, classes, scores, valid_detection])

            return model, prediction_model

        elif mode=="predict":
            bboxes, classes, scores, valid_detection = tf.keras.layers.Lambda(lambda x: decode(*x, max_objects=max_objects))([y1, y2, y3])
            prediction_model = tf.keras.Model(inputs=image_input, outputs=[bboxes, classes, scores, valid_detection])

            return prediction_model


def get_train_model(base_model, input_shape, num_classes, max_objects=100):
    output_size = input_shape[0] // 4
    hm_input = tf.keras.Input(shape=(output_size, output_size, num_classes))
    wh_input = tf.keras.Input(shape=(max_objects, 2))
    reg_input = tf.keras.Input(shape=(max_objects, 2))
    reg_mask_input = tf.keras.Input(shape=(max_objects,))
    index_input = tf.keras.Input(shape=(max_objects,))

    y1, y2, y3 = base_model.output
    loss_ = tf.keras.layers.Lambda(loss, output_shape = (1, ),name='centernet_loss')([y1, y2, y3, hm_input, wh_input, reg_input, reg_mask_input, index_input])
    model = tf.keras.Model(inputs=[base_model.input, hm_input, wh_input, reg_input, reg_mask_input, index_input], outputs=[loss_])
        
    return model
