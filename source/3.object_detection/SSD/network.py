import os
import numpy as np
import tensorflow as tf

def create_vgg16_layers():
    vgg16_conv4 = [tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
                   tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
                   tf.keras.layers.MaxPool2D(2, 2, padding='same'),
                   tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
                   tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
                   tf.keras.layers.MaxPool2D(2, 2, padding='same'),

                   tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu'),
                   tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu'),
                   tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu'),
                   tf.keras.layers.MaxPool2D(2, 2, padding='same'),

                   tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu'),
                   tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu'),
                   tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu'),
                   tf.keras.layers.MaxPool2D(2, 2, padding='same'),

                   tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu'),
                   tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu'),
                   tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu'),]

    x = tf.keras.layers.Input(shape=[None, None, 3])
    out = x
    for layer in vgg16_conv4:
        out = layer(out)

    vgg16_conv4 = tf.keras.Model(x, out)

    vgg16_conv7 = [tf.keras.layers.MaxPool2D(3, 1, padding='same'),
                   tf.keras.layers.Conv2D(1024, 3, padding='same', dilation_rate=6, activation='relu'),
                   tf.keras.layers.Conv2D(1024, 1, padding='same', activation='relu'),]

    x = tf.keras.layers.Input(shape=[None, None, 512])
    out = x
    for layer in vgg16_conv7:
        out = layer(out)

    vgg16_conv7 = tf.keras.Model(x, out)

    return vgg16_conv4, vgg16_conv7

def create_extra_layers():
    extra_layers = [
        # 8th block output shape: B, 512, 10, 10
        tf.keras.Sequential([tf.keras.layers.Conv2D(256, 1, activation='relu'),
                    tf.keras.layers.Conv2D(512, 3, strides=2, padding='same', activation='relu'),]),
        
        # 9th block output shape: B, 256, 5, 5
        tf.keras.Sequential([tf.keras.layers.Conv2D(128, 1, activation='relu'),
                    tf.keras.layers.Conv2D(256, 3, strides=2, padding='same', activation='relu'),]),
        
        # 10th block output shape: B, 256, 3, 3
        tf.keras.Sequential([tf.keras.layers.Conv2D(128, 1, activation='relu'),
                    tf.keras.layers.Conv2D(256, 3, activation='relu'),]),

        # 11th block output shape: B, 256, 1, 1
        tf.keras.Sequential([tf.keras.layers.Conv2D(128, 1, activation='relu'),
                    tf.keras.layers.Conv2D(256, 3, activation='relu'),]),

        # 12th block output shape: B, 256, 1, 1
        tf.keras.Sequential([tf.keras.layers.Conv2D(128, 1, activation='relu'),
                    tf.keras.layers.Conv2D(256, 4, activation='relu'),])
    ]

    return extra_layers

def create_conf_head_layers(num_classes):
    conf_head_layers = [tf.keras.layers.Conv2D(4 * num_classes, kernel_size=3, padding='same'),  # for 4th block
                        tf.keras.layers.Conv2D(6 * num_classes, kernel_size=3, padding='same'),  # for 7th block
                        tf.keras.layers.Conv2D(6 * num_classes, kernel_size=3, padding='same'),  # for 8th block
                        tf.keras.layers.Conv2D(6 * num_classes, kernel_size=3, padding='same'),  # for 9th block
                        tf.keras.layers.Conv2D(4 * num_classes, kernel_size=3, padding='same'),  # for 10th block
                        tf.keras.layers.Conv2D(4 * num_classes, kernel_size=3, padding='same'),  # for 11th block
                        tf.keras.layers.Conv2D(4 * num_classes, kernel_size=1)  # for 12th block
    ]
    return conf_head_layers


def create_loc_head_layers():
    """ Create layers for regression """
    loc_head_layers = [tf.keras.layers.Conv2D(4 * 4, kernel_size=3, padding='same'),
                       tf.keras.layers.Conv2D(6 * 4, kernel_size=3, padding='same'),
                       tf.keras.layers.Conv2D(6 * 4, kernel_size=3, padding='same'),
                       tf.keras.layers.Conv2D(6 * 4, kernel_size=3, padding='same'),
                       tf.keras.layers.Conv2D(4 * 4, kernel_size=3, padding='same'),
                       tf.keras.layers.Conv2D(4 * 4, kernel_size=3, padding='same'),
                       tf.keras.layers.Conv2D(4 * 4, kernel_size=1)]

    return loc_head_layers


class SSD(tf.keras.Model):
    def __init__(self, num_classes, arch='ssd300'):
        super(SSD, self).__init__()
        self.num_classes = num_classes
        self.vgg16_conv4, self.vgg16_conv7 = create_vgg16_layers()
        self.batch_norm = tf.keras.layers.BatchNormalization(beta_initializer="glorot_uniform", gamma_initializer="glorot_uniform")
        self.extra_layers = create_extra_layers()
        self.conf_head_layers = create_conf_head_layers(num_classes)
        self.loc_head_layers = create_loc_head_layers()

        if arch == 'ssd300':
            self.extra_layers.pop(-1)
            self.conf_head_layers.pop(-2)
            self.loc_head_layers.pop(-2)

    def compute_heads(self, x, idx):
        conf = self.conf_head_layers[idx](x)
        conf = tf.reshape(conf, [conf.shape[0], -1, self.num_classes])

        loc = self.loc_head_layers[idx](x)
        loc = tf.reshape(loc, [loc.shape[0], -1, 4])

        return conf, loc

    def init_vgg16(self):
        origin_vgg = tf.keras.applications.vgg16.VGG16(weights='imagenet')
        
        for i in range(len(self.vgg16_conv4.layers)):
            self.vgg16_conv4.get_layer(index=i).set_weights(
                origin_vgg.get_layer(index=i).get_weights())

        fc1_weights, fc1_biases = origin_vgg.get_layer(index=-3).get_weights()
        fc2_weights, fc2_biases = origin_vgg.get_layer(index=-2).get_weights()

        conv6_weights = np.random.choice(np.reshape(fc1_weights, (-1,)), (3, 3, 512, 1024))
        conv6_biases = np.random.choice(fc1_biases, (1024,))

        conv7_weights = np.random.choice(np.reshape(fc2_weights, (-1,)), (1, 1, 1024, 1024))
        conv7_biases = np.random.choice(fc2_biases, (1024,))

        self.vgg16_conv7.get_layer(index=2).set_weights([conv6_weights, conv6_biases])
        self.vgg16_conv7.get_layer(index=3).set_weights([conv7_weights, conv7_biases])

    def call(self, x):
        confs = []
        locs = []
        head_idx = 0
        for i in range(len(self.vgg16_conv4.layers)):
            x = self.vgg16_conv4.get_layer(index=i)(x)
            if i == len(self.vgg16_conv4.layers) - 5:
                conf, loc = self.compute_heads(self.batch_norm(x), head_idx)
                confs.append(conf)
                locs.append(loc)
                head_idx += 1

        x = self.vgg16_conv7(x)

        conf, loc = self.compute_heads(x, head_idx)

        confs.append(conf)
        locs.append(loc)
        head_idx += 1

        for layer in self.extra_layers:
            x = layer(x)
            conf, loc = self.compute_heads(x, head_idx)
            confs.append(conf)
            locs.append(loc)
            head_idx += 1

        confs = tf.concat(confs, axis=1)
        locs = tf.concat(locs, axis=1)

        return confs, locs
    
def create_ssd(num_classes, arch, pretrained_type, checkpoint_dir=None, checkpoint_path=None):
    net = SSD(num_classes, arch)
    net(tf.random.normal((1, 512, 512, 3)))
    
    if pretrained_type == 'base':
        net.init_vgg16()
    elif pretrained_type == 'latest':
        try:
            paths = [os.path.join(checkpoint_dir, path)
                     for path in os.listdir(checkpoint_dir)]
            latest = sorted(paths, key=os.path.getmtime)[-1]
            net.load_weights(latest)

        except AttributeError as e:
            print('Please make sure there is at least one checkpoint at {}'.format(
                checkpoint_dir))
            print('The model will be loaded from base weights.')
            net.init_vgg16()

        except ValueError as e:
            raise ValueError('Please check the following\n1./ Is the path correct: {}?\n2./ Is the model architecture correct: {}?'.format(latest, arch))
        
        except Exception as e:
            print(e)
            raise ValueError('Please check if checkpoint_dir is specified')

    elif pretrained_type == 'specified':
        if not os.path.isfile(checkpoint_path):
            raise ValueError(
                'Not a valid checkpoint file: {}'.format(checkpoint_path))

        try:
            net.load_weights(checkpoint_path)

        except Exception as e:
            raise ValueError('Please check the following\n1./ Is the path correct: {}?\n2./ Is the model architecture correct: {}?'.format(checkpoint_path, arch))
            
    else:
        raise ValueError('Unknown pretrained type: {}'.format(pretrained_type))

    return net