import tensorflow as tf


class BasicBlock(tf.keras.layers.Layer):
    def __init__(self, filters, strides=1):
        super(BasicBlock, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), strides=strides, padding="SAME")
        self.bn1 = tf.keras.layers.BatchNormalization()
        
        self.conv2 = tf.keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), strides=1, padding="SAME")
        self.bn2 = tf.keras.layers.BatchNormalization()

        if strides != 1:
            self.downsample = tf.keras.Sequential()
            self.downsample.add(tf.keras.layers.Conv2D(filters=filters, kernel_size=(1, 1), strides=strides))
            self.downsample.add(tf.keras.layers.BatchNormalization())
        else:
            self.downsample = lambda x : x

    def call(self, inputs, training=None, **kwargs):
        residual = self.downsample(inputs)
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)

        output = tf.nn.relu(tf.keras.layers.add([residual, x]))

        return output


def make_basic_block_layer(filters, blocks, strides=1):
    res_block = tf.keras.Sequential()
    res_block.add(BasicBlock(filters, strides))

    for _ in range(1, blocks):
        res_block.add(BasicBlock(filters, strides=1))

    return res_block


class ResNet(tf.keras.Model):
    def __init__(self, layer_params):
        super(ResNet, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=1, padding="SAME")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=1, padding="SAME")

        self.layer1 = make_basic_block_layer(filters=64, blocks=layer_params[0])
        self.layer2 = make_basic_block_layer(filters=128, blocks=layer_params[1], strides=2)
        self.layer3 = make_basic_block_layer(filters=256, blocks=layer_params[2], strides=2)
        self.layer4 = make_basic_block_layer(filters=512, blocks=layer_params[3], strides=2)


    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool1(x)
        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)

        return x


def resnet_backbone():
    return ResNet(layer_params=[2, 2, 2, 2])


if __name__ == "__main__":
    model = resnet_backbone()
    model.build(input_shape=(None, 120, 160, 1))
    model.summary()