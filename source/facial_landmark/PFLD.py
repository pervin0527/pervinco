import losses
import tensorflow as tf

def _inverted_res_block(inputs, expansion, stride, filters, use_res_connect, stage=1, block_id=1, expand=True, output2=False):
    in_channels = tf.keras.backend.int_shape(inputs)[-1]
    name = 'bbn_stage{}_block{}'.format(stage, block_id)

    x = tf.keras.layers.Conv2D(expansion*in_channels, kernel_size=1, padding='same', use_bias=False, name=name + '_expand_conv')(inputs)
    x = tf.keras.layers.BatchNormalization(epsilon=1e-3, momentum=0.999, name=name + '_expand_bn')(x)
    x = tf.keras.layers.ReLU(name=name + 'expand_relu')(x)

    x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=stride, use_bias=False, padding='same', name=name+'_dw_conv')(x)
    x = tf.keras.layers.BatchNormalization(epsilon=1e-3, momentum=0.999, name=name+'_dw_bn')(x)
    x = tf.keras.layers.ReLU(name=name + '_dw_relu')(x)

    x = tf.keras.layers.Conv2D(filters, kernel_size=1, padding='same', activation=None, use_bias=False, name=name + '_project_conv')(x)
    x = tf.keras.layers.BatchNormalization(epsilon=1e-3, momentum=0.999, name=name + '_project_bn')(x)

    if use_res_connect:
        return tf.keras.layers.Add(name=name+'_add')([inputs, x])
        # return input + x
    
    return x


def create_pfld_inference(input_size):
    image_input = tf.keras.layers.Input(shape=(input_size, input_size, 3))
    
    ## 112, 112, 3 ----> 56, 56, 64
    x = tf.keras.layers.Conv2D(64, kernel_size=3, strides=2, padding='same', use_bias=False, name='conv1')(image_input)
    x = tf.keras.layers.BatchNormalization(name='conv1_bn')(x)
    x = tf.keras.layers.ReLU(name='conv1_relu')(x)

    ## 56, 56, 64 ----> 56, 56, 64
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, padding="same")(x)
    # x = tf.keras.layers.Conv2D(64, kernel_size=3, strides=1, padding='same', use_bias=False, name='conv2')(x)
    x = tf.keras.layers.BatchNormalization(name='conv2_bn')(x)
    x = tf.keras.layers.ReLU(name='conv2_relu')(x)
    
    ## 56, 56, 64 ----> 28, 28, 64
    ## repeated 5, stride first layer : 2, expansion : 2
    x = _inverted_res_block(x, expansion=2, stride=2, filters=64, use_res_connect=False, stage=3, block_id=1)
    x = _inverted_res_block(x, expansion=2, stride=1, filters=64, use_res_connect=True, stage=3, block_id=2)
    x = _inverted_res_block(x, expansion=2, stride=1, filters=64, use_res_connect=True, stage=3, block_id=3)
    x = _inverted_res_block(x, expansion=2, stride=1, filters=64, use_res_connect=True, stage=3, block_id=4)
    out1 = _inverted_res_block(x, expansion=2, stride=1, filters=64, use_res_connect=True, stage=3, block_id=5)

    ## 28, 28, 64 ----> 14, 14, 128    
    x = _inverted_res_block(out1, expansion=2, stride=2, filters=128, use_res_connect=False, stage=4, block_id=1)
    
    ## 14, 14, 128 ----> 14, 14, 128
    x = _inverted_res_block(x, expansion=2, stride=1, filters=128, use_res_connect=False, stage=5, block_id=1)
    x = _inverted_res_block(x, expansion=4, stride=1, filters=128, use_res_connect=True, stage=5, block_id=2)
    x = _inverted_res_block(x, expansion=4, stride=1, filters=128, use_res_connect=True, stage=5, block_id=3)
    x = _inverted_res_block(x, expansion=4, stride=1, filters=128, use_res_connect=True, stage=5, block_id=4)
    x = _inverted_res_block(x, expansion=4, stride=1, filters=128, use_res_connect=True, stage=5, block_id=5)
    x = _inverted_res_block(x, expansion=4, stride=1, filters=128, use_res_connect=True, stage=5, block_id=6)

    ## 14, 14, 128 ----> 14, 14, 16
    x = _inverted_res_block(x, expansion=2, stride=1, filters=16, use_res_connect=False, stage=6, block_id=1)

    ## S1 : 14, 14, 16
    x1 = tf.keras.layers.AvgPool2D(14)(x)
    x1 = tf.keras.layers.Flatten()(x1)

    ## S2 : 14, 14, 16 ----> 7, 7, 32
    x = tf.keras.layers.Conv2D(32, kernel_size=3, strides=2, padding='same', use_bias=False, name='conv7')(x)
    x = tf.keras.layers.BatchNormalization(name='conv7_bn')(x)
    x = tf.keras.layers.ReLU(name='conv7_relu')(x)

    x2 = tf.keras.layers.AvgPool2D(7)(x)
    x2 = tf.keras.layers.Flatten()(x2)

    ## S3 : 7, 7, 32 ----> 1, 1, 128
    x = tf.keras.layers.Conv2D(128, kernel_size=7, strides=1, padding='valid', use_bias=False, name='conv_8')(x)
    x = tf.keras.layers.BatchNormalization(name='conv8_bn')(x)
    x = tf.keras.layers.ReLU(name='conv8_relu')(x)

    x3 = tf.keras.layers.Flatten()(x)
    
    multi_scale = tf.keras.layers.concatenate([x1, x2, x3], axis=1)
    landmarks = tf.keras.layers.Dense(196)(multi_scale)
    
    model = tf.keras.models.Model(inputs=image_input, outputs=[out1, landmarks])
    return model
    

def create_auxiliarynet(input_size):
    input = tf.keras.layers.Input(shape=(input_size, input_size, 64))

    ## 28, 28, 64 ---> 14, 14, 128    
    x = tf.keras.layers.Conv2D(128, kernel_size=3, strides=2, padding='same', use_bias=False, name='conv_1')(input)
    x = tf.keras.layers.BatchNormalization(name='conv1_bn')(x)
    x = tf.keras.layers.ReLU(name='conv1_relu')(x)

    ## 14, 14, 128 ---> 14, 14, 128
    x = tf.keras.layers.Conv2D(128, kernel_size=3, strides=1, padding='same', use_bias=False, name='conv_2')(x)
    x = tf.keras.layers.BatchNormalization(name='conv2_bn')(x)
    x = tf.keras.layers.ReLU(name='conv2_relu')(x)

    ## 14, 14, 128 ---> 7, 7, 32
    x = tf.keras.layers.Conv2D(32, kernel_size=3, strides=2, padding='same', use_bias=False, name='conv_3')(x)
    x = tf.keras.layers.BatchNormalization(name='conv3_bn')(x)
    x = tf.keras.layers.ReLU(name='conv3_relu')(x)

    ## 7, 7, 32 ----> 3, 3, 128
    x = tf.keras.layers.Conv2D(128, kernel_size=7, strides=1, padding='same', use_bias=False, name='conv_4')(x)
    x = tf.keras.layers.BatchNormalization(name='conv4_bn')(x)
    x = tf.keras.layers.ReLU(name='conv4_relu')(x)

    x = tf.keras.layers.MaxPool2D(3)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(32)(x)
    x = tf.keras.layers.Dense(3)(x)

    model = tf.keras.models.Model(inputs=input, outputs=x)

    return model


class PFLD(tf.keras.Model):
    def __init__(self, input_size=112, summary=False) -> None:
        super(PFLD, self).__init__()
        self.pfld_inference = create_pfld_inference(input_size=input_size)
        self.auxiliarynet = create_auxiliarynet(input_size=28)

        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.loss_tracker_2 = tf.keras.metrics.Mean(name="weighted_loss")

        if summary:
            self.pfld_inference.summary()
            self.auxiliarynet.summary()
    
    def call(self, x, training=False):
        features, landmarks = self.pfld_inference(x)
        if training:
            angle = self.auxiliarynet(features)
            return angle, landmarks
        else:
            return landmarks

    def train_step(self, data):
        img_tensor, attribute_gt, landmark_gt, euler_angle_gt = data
        
        with tf.GradientTape() as tape:
            angle, landmarks = self(img_tensor, training=True)
            weighted_loss, loss = losses.loss_fn(attribute_gt, landmark_gt, euler_angle_gt, angle, landmarks)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(weighted_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.loss_tracker.update_state(loss)
        self.loss_tracker_2.update_state(weighted_loss)
        return {"loss": self.loss_tracker.result(), "weighted_loss": self.loss_tracker_2.result()}

    def test_step(self, data):
        img_tensor, _, landmark_gt, _ = data
        
        landmarks = self(img_tensor, training=False)
        loss = tf.reduce_mean(tf.reduce_sum((landmark_gt - landmarks) * (landmark_gt - landmarks)))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    @property
    def metrics(self):
        return [self.loss_tracker, self.loss_tracker_2]


class PFLD_wing_loss_fn(tf.keras.Model):
    def __init__(self, input_size=112, summary=False) -> None:
        super(PFLD_wing_loss_fn, self).__init__()
        self.pfld_inference = create_pfld_inference(input_size=input_size)

        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

        if summary:
            self.pfld_inference.summary()
        

    def call(self, x):
        _, landmarks = self.pfld_inference(x)
        return landmarks

    def train_step(self, data):
        img_tensor, _, landmark_gt, _ = data
        
        with tf.GradientTape() as tape:
            landmarks = self(img_tensor, training=True)

            loss = losses.wing_loss(landmark_gt, landmarks)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        img_tensor, _, landmark_gt, _ = data
        
        landmarks = self(img_tensor, training=False)
        
        loss = tf.reduce_mean(tf.reduce_sum((landmark_gt - landmarks) * (landmark_gt - landmarks)))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    @property
    def metrics(self):
        return [self.loss_tracker]