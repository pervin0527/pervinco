from matplotlib import image
import tensorflow as tf
from functools import wraps
from utils.utils import compose
from loss import get_yolo_loss
from tensorflow.keras import backend as K
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.regularizers import L2

class SiLU(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(SiLU, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        return inputs * K.sigmoid(inputs)

    def get_config(self):
        config = super(SiLU, self).get_config()
        return config

    def compute_output_shape(self, input_shape):
        return input_shape


class Focus(tf.keras.layers.Layer):
    def __init__(self):
        super(Focus, self).__init__()

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] // 2 if input_shape[1] != None else input_shape[1], input_shape[2] // 2 if input_shape[2] != None else input_shape[2], input_shape[3] * 4)

    def call(self, x):
        return tf.concat([x[...,  ::2,  ::2, :],
                          x[..., 1::2,  ::2, :],
                          x[...,  ::2, 1::2, :],
                          x[..., 1::2, 1::2, :]], axis=-1)

# @wraps(tf.keras.layers.Conv2D)
def DarknetConv2D(filters, 
                  kernel_size,
                  strides=(1, 1),
                  padding="VALID",
                  kernel_initializer="glorot_uniform",
                  kernel_regularizer=None,
                  use_bias=False):

    if strides == (2, 2):
        padding = "VALID"
    else:
        padding = "SAME"

    return tf.keras.layers.Conv2D(filters=filters,
                                  kernel_size=kernel_size,
                                  strides=strides,
                                  padding=padding,
                                  kernel_initializer=kernel_initializer,
                                  kernel_regularizer=kernel_regularizer,
                                  use_bias=use_bias)

def DarknetConv2D_BN_SiLU(inputs,
                          filters, 
                          kernel_size, 
                          strides=(1, 1), 
                          padding="VALID", 
                          kernel_initializer="glorot_uniform", 
                          kernel_regularizer=None):

    x = DarknetConv2D(filters, kernel_size, strides, padding, kernel_initializer, kernel_regularizer)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = SiLU()(x)

    return x

def SPPBottleneck(x, out_channels, weight_decay=5e-4, name = ""):
    x = DarknetConv2D_BN_SiLU(x, out_channels // 2, (1, 1))
    maxpool1 = tf.keras.layers.MaxPooling2D(pool_size=(5, 5), strides=(1, 1), padding='same')(x)
    maxpool2 = tf.keras.layers.MaxPooling2D(pool_size=(9, 9), strides=(1, 1), padding='same')(x)
    maxpool3 = tf.keras.layers.MaxPooling2D(pool_size=(13, 13), strides=(1, 1), padding='same')(x)
    x = tf.keras.layers.Concatenate()([x, maxpool1, maxpool2, maxpool3])
    x = DarknetConv2D_BN_SiLU(x, out_channels, (1, 1))

    return x

def Bottleneck(x, out_channels, shortcut=True, weight_decay=5e-4, name = ""):
    y = DarknetConv2D_BN_SiLU(x, out_channels, (1, 1))
    y = DarknetConv2D_BN_SiLU(y, out_channels, (3, 3))

    if shortcut:
        y = tf.keras.layers.Add()([x, y])

    return y

def CSPLayer(x, num_filters, num_blocks, shortcut=True, expansion=0.5, weight_decay=5e-4, name=""):
    hidden_channels = int(num_filters * expansion)
    x_1 = DarknetConv2D_BN_SiLU(x, hidden_channels, (1, 1))

    x_2 = DarknetConv2D_BN_SiLU(x, hidden_channels, (1, 1))
    for i in range(num_blocks):
        x_1 = Bottleneck(x_1, hidden_channels, shortcut=shortcut, weight_decay=weight_decay, name = name + '.m.' + str(i))

    route = tf.keras.layers.Concatenate()([x_1, x_2])

    return DarknetConv2D_BN_SiLU(route, num_filters, (1, 1))


def resblock_body(x, num_filters, num_blocks, expansion=0.5, shortcut=True, last=False, weight_decay=5e-4, name = ""):
    x = tf.keras.layers.ZeroPadding2D(((1, 0),(1, 0)))(x)
    x = DarknetConv2D_BN_SiLU(x, num_filters, (3, 3), strides = (2, 2))
    if last:
        x = SPPBottleneck(x, num_filters, weight_decay=weight_decay, name = name + '.1')
    return CSPLayer(x, num_filters, num_blocks, shortcut=shortcut, expansion=expansion, weight_decay=weight_decay, name = name + '.1' if not last else name + '.2')


def darknet_body(x, dep_mul, wid_mul, weight_decay=5e-4):
    base_channels   = int(wid_mul * 64)
    base_depth      = max(round(dep_mul * 3), 1)

    x = Focus()(x)
    x = DarknetConv2D_BN_SiLU(x, base_channels, (3, 3))
    x = resblock_body(x, base_channels * 2, base_depth, weight_decay=weight_decay, name = 'backbone.backbone.dark2')
    x = resblock_body(x, base_channels * 4, base_depth * 3, weight_decay=weight_decay, name = 'backbone.backbone.dark3')
    
    feat1 = x
    x = resblock_body(x, base_channels * 8, base_depth * 3, weight_decay=weight_decay, name = 'backbone.backbone.dark4')
    feat2 = x
    x = resblock_body(x, base_channels * 16, base_depth, shortcut=False, last=True, weight_decay=weight_decay, name = 'backbone.backbone.dark5')
    feat3 = x

    return feat1,feat2,feat3


def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image):
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = K.cast(input_shape, K.dtype(box_yx))
    image_shape = K.cast(image_shape, K.dtype(box_yx))

    if letterbox_image:
        new_shape = K.round(input_shape * K.min(input_shape/input_shape))
        offset  = (input_shape - new_shape)/2./input_shape
        scale   = input_shape/new_shape

        box_yx  = (box_yx - offset) * scale
        box_hw *= scale

    box_mins    = box_yx - (box_hw / 2.)
    box_maxes   = box_yx + (box_hw / 2.)
    boxes  = K.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]])
    boxes *= K.concatenate([input_shape, input_shape])
    return boxes


def DecodeBox(outputs,
              num_classes,
              input_shape,
              max_boxes = 100,
              confidence = 0.5,
              nms_iou = 0.3,
              letterbox_image = True):
            
    image_shape = K.reshape(outputs[-1], [-1])
    outputs     = outputs[:-1]

    bs      = K.shape(outputs[0])[0]
    grids   = []
    strides = []
    hw      = [K.shape(x)[1:3] for x in outputs]
    outputs = tf.concat([tf.reshape(x, [bs, -1, 5 + num_classes]) for x in outputs], axis = 1)
    for i in range(len(hw)):
        grid_x, grid_y  = tf.meshgrid(tf.range(hw[i][1]), tf.range(hw[i][0]))
        grid            = tf.reshape(tf.stack((grid_x, grid_y), 2), (1, -1, 2))
        shape           = tf.shape(grid)[:2]

        grids.append(tf.cast(grid, K.dtype(outputs)))
        strides.append(tf.ones((shape[0], shape[1], 1)) * input_shape[0] / tf.cast(hw[i][0], K.dtype(outputs)))
    grids               = tf.concat(grids, axis=1)
    strides             = tf.concat(strides, axis=1)
    box_xy = (outputs[..., :2] + grids) * strides / K.cast(input_shape[::-1], K.dtype(outputs))
    box_wh = tf.exp(outputs[..., 2:4]) * strides / K.cast(input_shape[::-1], K.dtype(outputs))

    box_confidence  = K.sigmoid(outputs[..., 4:5])
    box_class_probs = K.sigmoid(outputs[..., 5: ])
    boxes       = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image)
    box_scores  = box_confidence * box_class_probs

    mask             = box_scores >= confidence
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')
    boxes_out   = []
    scores_out  = []
    classes_out = []
    for c in range(num_classes):
        class_boxes      = tf.boolean_mask(boxes, mask[..., c])
        class_box_scores = tf.boolean_mask(box_scores[..., c], mask[..., c])
        nms_index = tf.image.non_max_suppression(class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=nms_iou)

        class_boxes         = K.gather(class_boxes, nms_index)
        class_box_scores    = K.gather(class_box_scores, nms_index)
        classes             = K.ones_like(class_box_scores, 'int32') * c

        boxes_out.append(class_boxes)
        scores_out.append(class_box_scores)
        classes_out.append(classes)
    boxes_out      = K.concatenate(boxes_out, axis=0)
    scores_out     = K.concatenate(scores_out, axis=0)
    classes_out    = K.concatenate(classes_out, axis=0)

    return boxes_out, scores_out, classes_out


def yolov3(input_shape, num_classes, phi, weight_decay=5e-4):
    depth_dict = {'tiny': 0.33, 's' : 0.33, 'm' : 0.67, 'l' : 1.00, 'x' : 1.33,}
    width_dict = {'tiny': 0.375, 's' : 0.50, 'm' : 0.75, 'l' : 1.00, 'x' : 1.25,}
    depth, width = depth_dict[phi], width_dict[phi]
    in_channels = [256, 512, 1024]

    inputs = tf.keras.Input(input_shape)
    feat1, feat2, feat3 = darknet_body(inputs, depth, width, weight_decay=weight_decay)

    P5 = DarknetConv2D_BN_SiLU(feat3, int(in_channels[1] * width), (1, 1))
    P5_upsample = tf.keras.layers.UpSampling2D()(P5)
    P5_upsample = tf.keras.layers.Concatenate(axis = -1)([P5_upsample, feat2])
    P5_upsample = CSPLayer(P5_upsample, int(in_channels[1] * width), round(3 * depth), shortcut = False)

    P4 = DarknetConv2D_BN_SiLU(P5_upsample, int(in_channels[0] * width), (1, 1))
    P4_upsample = tf.keras.layers.UpSampling2D()(P4)
    P4_upsample = tf.keras.layers.Concatenate(axis = -1)([P4_upsample, feat1])
    P3_out = CSPLayer(P4_upsample, int(in_channels[0] * width), round(3 * depth), shortcut = False, weight_decay=weight_decay, name = 'backbone.C3_p3')

    P3_downsample = tf.keras.layers.ZeroPadding2D(((1, 0),(1, 0)))(P3_out)
    P3_downsample = DarknetConv2D_BN_SiLU(P3_downsample, int(in_channels[0] * width), (3, 3), strides = (2, 2))
    P3_downsample = tf.keras.layers.Concatenate(axis = -1)([P3_downsample, P4])
    P4_out = CSPLayer(P3_downsample, int(in_channels[1] * width), round(3 * depth), shortcut = False, weight_decay=weight_decay, name = 'backbone.C3_n3')

    P4_downsample = tf.keras.layers.ZeroPadding2D(((1, 0),(1, 0)))(P4_out)
    P4_downsample = DarknetConv2D_BN_SiLU(P4_downsample, int(in_channels[1] * width), (3, 3), strides = (2, 2))
    P4_downsample = tf.keras.layers.Concatenate(axis = -1)([P4_downsample, P5])
    P5_out = CSPLayer(P4_downsample, int(in_channels[2] * width), round(3 * depth), shortcut = False, weight_decay=weight_decay, name = 'backbone.C3_n4')

    fpn_outs = [P3_out, P4_out, P5_out]
    yolo_outs = []
    for i, out in enumerate(fpn_outs):
        stem = DarknetConv2D_BN_SiLU(out, int(256 * width), (1, 1), strides = (1, 1))
        cls_conv = DarknetConv2D_BN_SiLU(stem, int(256 * width), (3, 3), strides = (1, 1))
        cls_conv = DarknetConv2D_BN_SiLU(cls_conv, int(256 * width), (3, 3), strides = (1, 1))

        cls_pred = DarknetConv2D(num_classes, (1, 1), strides = (1, 1))(cls_conv)

        reg_conv = DarknetConv2D_BN_SiLU(stem, int(256 * width), (3, 3), strides = (1, 1))
        reg_conv = DarknetConv2D_BN_SiLU(reg_conv, int(256 * width), (3, 3), strides = (1, 1))
        reg_pred = DarknetConv2D(4, (1, 1), strides = (1, 1))(reg_conv)

        obj_pred = DarknetConv2D(1, (1, 1), strides = (1, 1))(reg_conv)
        output = tf.keras.layers.Concatenate(axis = -1)([reg_pred, obj_pred, cls_pred])
        yolo_outs.append(output)

        model = tf.keras.Model(inputs, yolo_outs)

    return model

def get_train_model(base_model, input_shape, num_classes):
    coordinates = tf.keras.Input(shape=(None, 5))
    model_loss = tf.keras.layers.Lambda(get_yolo_loss(input_shape, len(base_model.output), num_classes), 
                                         output_shape    = (1, ), 
                                         name            = 'yolo_loss',)([*base_model.output, coordinates])
    
    model = tf.keras.Model([base_model.input, coordinates], model_loss)

    return model