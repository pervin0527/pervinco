import tensorflow as tf
import tensorflow.keras.backend as K

################################################################################
# Layers
################################################################################
class ConvolutionBnActivation(tf.keras.layers.Layer):
    """
    """
    # def __init__(self, filters, kernel_size, strides=(1, 1), activation=tf.keras.activations.relu, **kwargs):
    def __init__(self, filters, kernel_size, strides=(1, 1), padding="same", data_format=None, dilation_rate=(1, 1),
                 groups=1, activation=None, kernel_initializer="glorot_uniform", bias_initializer="zeros", kernel_regularizer=None,
                 bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, use_batchnorm=False, 
                 axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, trainable=True,
                 post_activation="relu", block_name=None):
        super(ConvolutionBnActivation, self).__init__()


        # 2D Convolution Arguments
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.data_format = data_format
        self.dilation_rate = dilation_rate
        self.activation = activation
        self.use_bias = not (use_batchnorm)
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint

        # Batch Normalization Arguments
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.trainable = trainable
        
        self.block_name = block_name
        
        self.conv = None
        self.bn = None
        #tf.keras.layers.BatchNormalization(scale=False, momentum=0.9)
        self.post_activation = tf.keras.layers.Activation(post_activation)

    def build(self, input_shape):
        self.conv = tf.keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
            activation=self.activation,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            activity_regularizer=self.activity_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint,
            name=self.block_name + "_conv" if self.block_name is not None else None

        )

        self.bn = tf.keras.layers.BatchNormalization(
            axis=self.axis,
            momentum=self.momentum,
            epsilon=self.epsilon,
            center=self.center,
            scale=self.scale,
            trainable=self.trainable,
            name=self.block_name + "_bn" if self.block_name is not None else None
        )

    def call(self, x, training=None):
        x = self.conv(x)
        x = self.bn(x, training=training)
        x = self.post_activation(x)

        return x

    def compute_output_shape(self, input_shape):
        print(input_shape)
        return [input_shape[0], input_shape[1], input_shape[2], self.filters]

class AtrousSeparableConvolutionBnReLU(tf.keras.layers.Layer):
    """
    """
    def __init__(self, filters, kernel_size, strides=[1, 1, 1, 1], padding="SAME", data_format=None,
                 dilation=None, channel_multiplier=1, axis=-1, momentum=0.99, epsilon=0.001,
                 center=True, scale=True, trainable=True, post_activation=None, block_name=None):
        super(AtrousSeparableConvolutionBnReLU, self).__init__()

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.data_format = data_format
        self.dilation = dilation
        self.channel_multiplier = channel_multiplier

        # Batch Normalization Arguments
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.trainable = trainable
        
        self.block_name = block_name
        
        self.bn = None

        self.activation = tf.keras.layers.Activation(tf.keras.activations.relu)
        
        self.dw_filter = None
        self.pw_filter = None

    def build(self, input_shape):
        in_channels = input_shape[-1]
        self.dw_filter = self.add_weight(
            name="dw_kernel",
            shape=[self.kernel_size, self.kernel_size, in_channels, self.channel_multiplier],
            initializer=tf.keras.initializers.GlorotNormal(),
            regularizer=tf.keras.regularizers.l2(l=1e-4),
            trainable=True
        )
        self.pw_filter = self.add_weight(
            name="pw_kernel",
            shape=[1, 1, in_channels * self.channel_multiplier, self.filters],
            initializer=tf.keras.initializers.GlorotNormal(),
            regularizer=tf.keras.regularizers.l2(l=1e-4),
            trainable=True
        )

        self.bn = tf.keras.layers.BatchNormalization(
            axis=self.axis,
            momentum=self.momentum,
            epsilon=self.epsilon,
            center=self.center,
            scale=self.scale,
            name=self.block_name + "_bn" if self.block_name is not None else None
        )
        
    def call(self, x, training=None):
        x = tf.nn.separable_conv2d(
            x,
            self.dw_filter,
            self.pw_filter,
            strides=self.strides,
            dilations=[self.dilation, self.dilation],
            padding=self.padding,
            )
        x = self.bn(x, training=training)
        x = self.activation(x)

        return x

    def compute_output_shape(self, input_shape):
        print(input_shape)
        return [input_shape[0], input_shape[1], input_shape[2], self.filters]

class AtrousSpatialPyramidPoolingV3(tf.keras.layers.Layer):
    """
    """
    def __init__(self, atrous_rates, filters):
        super(AtrousSpatialPyramidPoolingV3, self).__init__()
        self.filters = filters

        # adapt scale and mometum for bn
        self.conv_bn_relu = ConvolutionBnActivation(filters=filters, kernel_size=1)

        self.atrous_sepconv_bn_relu_1 = AtrousSeparableConvolutionBnReLU(dilation=atrous_rates[0], filters=filters, kernel_size=3)
        self.atrous_sepconv_bn_relu_2 = AtrousSeparableConvolutionBnReLU(dilation=atrous_rates[1], filters=filters, kernel_size=3)
        self.atrous_sepconv_bn_relu_3 = AtrousSeparableConvolutionBnReLU(dilation=atrous_rates[2], filters=filters, kernel_size=3)

        # 1x1 reduction convolutions
        self.conv_reduction_1 = tf.keras.layers.Conv2D(
            filters=256,
            kernel_size=1,
            use_bias=False,
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-4))


    def call(self, input_tensor, training=None):
        # global average pooling input_tensor
        glob_avg_pool = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=[1, 2], keepdims=True))(input_tensor)
        glob_avg_pool = self.conv_bn_relu(glob_avg_pool, training=training)
        glob_avg_pool = tf.keras.layers.Lambda(lambda x: tf.image.resize(x, [input_tensor.shape[1], input_tensor.shape[2]]))(glob_avg_pool)

        # process with atrous
        w = self.conv_bn_relu(input_tensor, training=training)
        x = self.atrous_sepconv_bn_relu_1(input_tensor, training=training)
        y = self.atrous_sepconv_bn_relu_2(input_tensor, training=training)
        z = self.atrous_sepconv_bn_relu_3(input_tensor, training=training)

        # concatenation
        net = tf.concat([glob_avg_pool, w, x, y, z], axis=-1)
        net = self.conv_reduction_1(net, training=training)

        return net

    def compute_output_shape(self, input_shape):
        print(input_shape)
        return [input_shape[0], input_shape[1], input_shape[2], 256]

################################################################################
# DeepLabV3+
################################################################################
class DeepLabV3plus(tf.keras.Model):
    def __init__(self, n_classes, base_model, output_layers, height=None, width=None, filters=256,
                 final_activation="softmax", backbone_trainable=False,
                 output_stride=8, dilations=[6, 12, 18], **kwargs):
        super(DeepLabV3plus, self).__init__(**kwargs)

        self.n_classes = n_classes
        self.backbone = None
        self.filters = filters
        self.final_activation = final_activation
        self.output_stride = output_stride
        self.dilations = dilations
        self.height = height
        self.width = width


        if self.output_stride == 8:
            self.upsampling2d_1 = tf.keras.layers.UpSampling2D(size=2, interpolation="bilinear")
            output_layers = output_layers[:3]
            self.dilations = [2 * rate for rate in dilations]
        elif self.output_stride == 16:
            self.upsampling2d_1 = tf.keras.layers.UpSampling2D(size=4, interpolation="bilinear")
            output_layers = output_layers[:4]
            self.dilations = dilations
        else:
            raise ValueError("'output_stride' must be one of (8, 16), got {}".format(self.output_stride))

        base_model.trainable = backbone_trainable
        self.backbone = tf.keras.Model(inputs=base_model.input, outputs=output_layers)

        # Define Layers
        self.atrous_sepconv_bn_relu_1 = AtrousSeparableConvolutionBnReLU(dilation=2, filters=filters, kernel_size=3)
        self.atrous_sepconv_bn_relu_2 = AtrousSeparableConvolutionBnReLU(dilation=2, filters=filters, kernel_size=3)
        self.aspp = AtrousSpatialPyramidPoolingV3(self.dilations, filters)
        
        self.conv1x1_bn_relu_1 = ConvolutionBnActivation(filters, 1)
        self.conv1x1_bn_relu_2 = ConvolutionBnActivation(64, 1)

        self.upsample2d_1 = tf.keras.layers.UpSampling2D(size=2, interpolation="bilinear")
        self.upsample2d_2 = tf.keras.layers.UpSampling2D(size=4, interpolation="bilinear")

        self.concat = tf.keras.layers.Concatenate(axis=3)
        
        self.conv3x3_bn_relu_1 = ConvolutionBnActivation(filters, 3)
        self.conv3x3_bn_relu_2 = ConvolutionBnActivation(filters, 3)
        self.conv1x1_bn_sigmoid = ConvolutionBnActivation(self.n_classes, 1, post_activation="linear")

        self.final_activation = tf.keras.layers.Activation(final_activation)

    def call(self, inputs, training=None, mask=None):
        if training is None:
            training = True

        x = self.backbone(inputs)[-1]
        low_level_features = self.backbone(inputs)[1]
        
        # Encoder Module
        encoder = self.atrous_sepconv_bn_relu_1(x, training)
        encoder = self.aspp(encoder, training)
        encoder = self.conv1x1_bn_relu_1(encoder, training)
        encoder = self.upsample2d_1(encoder)

        # Decoder Module
        decoder_low_level_features = self.atrous_sepconv_bn_relu_2(low_level_features, training)
        decoder_low_level_features = self.conv1x1_bn_relu_2(decoder_low_level_features, training)
        # decoder_low_level_features = tf.image.resize(decoder_low_level_features, (encoder.shape[1], encoder.shape[2]))

        decoder = self.concat([decoder_low_level_features, encoder])
        
        decoder = self.conv3x3_bn_relu_1(decoder, training)
        decoder = self.conv3x3_bn_relu_2(decoder, training)
        decoder = self.conv1x1_bn_sigmoid(decoder, training)

        decoder = self.upsample2d_2(decoder)
        decoder = self.final_activation(decoder)

        return decoder

    def model(self):
        x = tf.keras.layers.Input(shape=(self.height, self.width, 3))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))