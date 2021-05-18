import os, sys
import numpy as np
import tensorflow as tf

from transform_net import Input_Transform_Net, Feature_Transform_Net, ConvBN, FC

class ClsModel(tf.keras.Model):

    def __init__(self, num_points):
        super(ClsModel, self).__init__()
        self.num_points = num_points
        self.input_transform = Input_Transform_Net(self.num_points)
        self.feature_transform = Feature_Transform_Net(self.num_points, K=64)
        self.conv1 = ConvBN(64, [1, 3], padding='valid', strides=(1, 1), bn=True, name='conv1')
        # self.conv1 = tf.keras.layers.Conv2D(64, [1, 3], padding='valid', strides=(1, 1))
        self.conv2 = ConvBN(64, [1, 1], padding='valid', strides=(1, 1), bn=True, name='conv2')

        self.conv3 = ConvBN(64, [1, 1], padding='valid', strides=(1, 1), bn=True, name='conv3')
        self.conv4 = ConvBN(128, [1, 1], padding='valid', strides=(1, 1), bn=True, name='conv4')
        self.conv5 = ConvBN(1024, [1, 1], padding='valid', strides=(1, 1), bn=True, name='conv5')

        self.maxpooling = tf.keras.layers.MaxPool2D([self.num_points, 1], padding='valid')
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = FC(512, name='fc1')
        self.drop1 = tf.keras.layers.Dropout(0.5)
        self.fc2 = FC(256, name='fc2')
        self.drop2 = tf.keras.layers.Dropout(0.5)
        self.fc3 = FC(40, name='fc3', activation=False, bn=False)

    def call(self, inputs):
        # input_transform
        end_points = {}
        transform = self.input_transform(inputs)
        inputs = tf.matmul(inputs, transform)
        inputs= tf.expand_dims(inputs, -1)
        out = self.conv1(inputs)
        out = self.conv2(out)
        # feature_transform
        transform2 = self.feature_transform(out)
        end_points['transform'] = transform2
        out_transform = tf.matmul(tf.squeeze(out, axis=2), transform2)
        out_transform = tf.expand_dims(out_transform, axis=2)

        out_transform = self.conv3(out_transform)
        out_transform = self.conv4(out_transform)
        out_transform = self.conv5(out_transform)

        out_transform = self.maxpooling(out_transform)
        out_transform = self.flatten(out_transform)
        out_transform = self.fc1(out_transform)
        out_transform = self.drop1(out_transform)
        out_transform = self.fc2(out_transform)
        out_transform = self.drop2(out_transform)
        out_transform = self.fc3(out_transform)

        return tf.nn.softmax(out_transform)


# if __name__ == '__main__':

#     model = ClsModel(1024)
#     model.build(input_shape=(None, 1024, 3))