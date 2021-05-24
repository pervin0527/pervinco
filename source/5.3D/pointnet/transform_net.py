import numpy as np
import tensorflow as tf

class ConvBN(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 kernal_size,
                 padding,
                 strides,
                 name,
                 use_xavier=True,
                 stddev=1e-3,
                 weight_decay=0.0,
                 bn=False,
                 ):
        super(ConvBN, self).__init__(name=name)
        self.check = bn
        if use_xavier:
            self.conv2d = tf.keras.layers.Conv2D(filters,
                                                 kernel_size=kernal_size,
                                                 padding=padding,
                                                 strides=strides,
                                                 kernel_regularizer=tf.keras.regularizers.l2(weight_decay))
        else:
            self.conv2d = tf.keras.layers.Conv2D(filters,
                                                 kernal_size=kernal_size,
                                                 padding=padding,
                                                 strides=strides,
                                                 kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=stddev),
                                                 kernel_regularizer=tf.keras.regularizers.l2(weight_decay))

        self.bn = tf.keras.layers.BatchNormalization()
        self.activation = tf.keras.layers.Activation('relu')

    def call(self, inputs):

        output = self.conv2d(inputs)
        if self.check:
            output = self.bn(output)
        output = self.activation(output)
        return output

class FC(tf.keras.layers.Layer):

    def __init__(self,
                 outdim,
                 name,
                 activation=True,
                 bn=True):
        super(FC, self).__init__(name=name)
        self.check = bn
        self.activation = activation
        self.fc = tf.keras.layers.Dense(outdim)
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.Activation('relu')

    def call(self, inputs):

        out = self.fc(inputs)
        if self.check:
            out = self.bn(out)
        if self.activation:
            out = self.relu(out)
        return out

class Input_Transform_Net(tf.keras.Model):
    def __init__(self, num_points, K=3):
        super(Input_Transform_Net, self).__init__(name='input_transform_net')
        self.K = K
        self.num_points = num_points

        self.conv1 = ConvBN(64, [1, 3], padding='valid', strides=(1, 1), bn=True, name='iconv1')
        self.conv2 = ConvBN(128, [1, 1], padding='valid', strides=(1, 1), bn=True, name='iconv2')
        self.conv3 = ConvBN(256, [1, 1], padding='valid', strides=(1, 1), bn=True, name='iconv3')
        self.maxpooling = tf.keras.layers.MaxPool2D([self.num_points, 1], padding='valid')
        self.flatten = tf.keras.layers.Flatten()
        ## fully connected
        self.fc1 = FC(512, name='ifc1')
        self.fc2 = FC(256, name='ifc2')

        self.w = tf.Variable(initial_value=tf.zeros([256, 3 * self.K]), dtype=tf.float32)
        self.b = tf.Variable(initial_value=tf.zeros([3 * self.K]), dtype=tf.float32)
        self.b.assign_add(tf.constant([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=tf.float32))


    def call(self, inputs, training=None):

        inputs = tf.expand_dims(inputs, -1)
        out = self.conv1(inputs)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.maxpooling(out)

        out = self.flatten(out)
        out = self.fc1(out)
        out = self.fc2(out)

        out = tf.matmul(out, self.w) + self.b
        transform = tf.reshape(out, [-1, 3, self.K])
        return transform


class Feature_Transform_Net(tf.keras.Model):

    def __init__(self, num_points, K=3):
        super(Feature_Transform_Net, self).__init__(name='feature_transform_net')
        self.K = K
        self.num_points = num_points
        self.conv1 = ConvBN(64, [1, 1], padding='valid', strides=(1, 1), bn=True, name='fconv1')
        self.conv2 = ConvBN(128, [1, 1], padding='valid', strides=(1, 1), bn=True, name='fconv2')
        self.conv3 = ConvBN(1024, [1, 1], padding='valid', strides=(1, 1), bn=True, name='fconv3')
        self.maxpooling = tf.keras.layers.MaxPool2D([self.num_points, 1], padding='valid')
        self.flatten = tf.keras.layers.Flatten()
        ## fully connected
        self.fc1 = FC(512, name='ffc1')
        self.fc2 = FC(256, name='ffc2')

        self.w = tf.Variable(initial_value=tf.zeros([256, self.K * self.K]), dtype=tf.float32)
        self.b = tf.Variable(initial_value=tf.zeros([self.K * self.K]), dtype=tf.float32)
        self.b.assign_add(tf.constant(np.eye(K).flatten(), dtype=tf.float32))


    def call(self, inputs, training=None):
            out = self.conv1(inputs)
            out = self.conv2(out)
            out = self.conv3(out)
            out = self.maxpooling(out)

            out = self.flatten(out)
            out = self.fc1(out)
            out = self.fc2(out)

            out = out@self.w + self.b
            transform = tf.reshape(out, [-1, self.K, self.K])
            return transform


# if __name__ == '__main__':
#     print("test..")
#     model = Feature_Transform_Net(1024, 64)
#     model.build(input_shape=(None, 1024, 1, 64))