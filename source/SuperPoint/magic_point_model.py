import tensorflow as tf
from model.loss import detector_loss
from model.resnet import resnet_backbone   


def vgg_block(inputs, filters, kernel_size, padding="SAME", strides=1, kernel_reg=0.0, activation=tf.nn.relu, batch_normalization=True):
    x = tf.keras.layers.Conv2D(filters=filters,
                               kernel_size=kernel_size,
                               padding=padding,
                               strides=strides,
                               activation=activation,
                               kernel_regularizer=tf.keras.regularizers.L2(kernel_reg))(inputs)
    
    if batch_normalization:
        x = tf.keras.layers.BatchNormalization()(x)

    return x


def vgg_backbone(inputs):
    inputs = tf.keras.Input(shape=inputs, name="image")
    x = vgg_block(inputs, filters=64, kernel_size=3, padding="SAME", strides=1, activation=tf.nn.relu)     ## 120, 160, 64
    x = vgg_block(x, filters=64, kernel_size=3, padding="SAME", strides=1, activation=tf.nn.relu)          ## 120, 160, 64
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding="SAME")(x)                          ## 60, 80, 64

    x = vgg_block(x, filters=64, kernel_size=3, padding="SAME", strides=1, activation=tf.nn.relu)          ## 60, 80, 64
    x = vgg_block(x, filters=64, kernel_size=3, padding="SAME", strides=1, activation=tf.nn.relu)          ## 60, 80, 64
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding="SAME")(x)                          ## 30, 40, 64

    x = vgg_block(x, filters=128, kernel_size=3, padding="SAME", strides=1, activation=tf.nn.relu)         ## 30, 40, 128
    x = vgg_block(x, filters=128, kernel_size=3, padding="SAME", strides=1, activation=tf.nn.relu)         ## 30, 40, 128
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding="SAME")(x)                          ## 15, 20, 128

    x = vgg_block(x, filters=128, kernel_size=3, padding="SAME", strides=1, activation=tf.nn.relu)
    output = vgg_block(x, filters=128, kernel_size=3, padding="SAME", strides=1, activation=tf.nn.relu)

    model = tf.keras.Model(inputs=inputs, outputs=output)
    return model


def detector_head(inputs, nms_size, threshold):                  
    inputs = tf.keras.Input(shape=inputs)
    x = vgg_block(inputs, filters=256, kernel_size=3, padding="SAME", strides=1, activation=tf.nn.relu)
    x = vgg_block(x, filters=1 + pow(8, 2), kernel_size=1, padding="VALID", strides=1, activation=None)

    prob = tf.keras.activations.softmax(x, axis=-1)
    prob = prob[:, :, :, :-1]
    prob = tf.nn.depth_to_space(prob, 8, data_format='NHWC')
    prob = tf.squeeze(prob, axis=-1)

    # prob = detector_postprocess()(prob, nms_size, threshold)

    model = tf.keras.Model(inputs=inputs, outputs=[x, prob])

    return model


class detector_postprocess(tf.keras.layers.Layer):
    @staticmethod
    def box_nms(prob, size, iou=0.1, threshold=0.01, keep_top_k=0):
        pts = tf.cast(tf.where(tf.greater_equal(prob, threshold)), dtype=tf.float32)
        size = tf.constant(size/2.)
        boxes = tf.concat([pts-size, pts+size], axis=1)
        scores = tf.gather_nd(prob, tf.cast(pts, dtype=tf.int32))
        
        indices = tf.image.non_max_suppression(boxes, scores, tf.shape(boxes)[0], iou)
        pts = tf.gather(pts, indices)
        scores = tf.gather(scores, indices)
        if keep_top_k:
            k = tf.minimum(tf.shape(scores)[0], tf.constant(keep_top_k))  # when fewer
            scores, indices = tf.nn.top_k(scores, k)
            pts = tf.gather(pts, indices)
        prob = tf.scatter_nd(tf.cast(pts, tf.int32), scores, tf.shape(prob))
        
        return prob

    def call(self, pred_prob, size, threshold):
        return tf.map_fn(lambda p : self.box_nms(p, size, threshold=threshold), pred_prob)


class MagicPoint(tf.keras.Model):
    def __init__(self, backbone_name, backbone_input, nms_size, threshold, summary=False):
        super(MagicPoint, self).__init__()

        if backbone_name.lower() == "vgg":
            self.backbone = vgg_backbone(inputs=(backbone_input))
            self.output_channel = 128

        elif backbone_name.lower() == "resnet":
            self.backbone = resnet_backbone()
            self.backbone.build(input_shape=(None, backbone_input[0], backbone_input[1], 1))
            self.output_channel = 512

        self.detector_head = detector_head((int(backbone_input[0] / 8), int(backbone_input[1] / 8), self.output_channel), nms_size, threshold)
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

        if summary:
            self.backbone.summary()
            self.detector_head.summary()


    def call(self, x, training=False):
        backbone_output = self.backbone(x)
        logits, prob = self.detector_head(backbone_output)

        return logits, prob


    def train_step(self, data):
        image, keypoints, valid_mask, keypoint_map = data["image"], data["keypoints"], data["valid_mask"], data["keypoint_map"]

        with tf.GradientTape() as tape:
            pred_logits, pred_prob = self(image, training=True)
            loss = detector_loss(keypoint_map, pred_logits, valid_mask)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.loss_tracker.update_state(loss)

        return {"loss" : self.loss_tracker.result()}


    def test_step(self, data):
        image, keypoints, valid_mask, keypoint_map = data["image"], data["keypoints"], data["valid_mask"], data["keypoint_map"]
        pred_logits, pred_prob = self(image, training=False)
        loss = detector_loss(keypoint_map, pred_logits, valid_mask)
        self.loss_tracker.update_state(loss)

        return {"loss" : self.loss_tracker.result()}