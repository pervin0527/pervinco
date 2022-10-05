from absl import app, flags, logging
from absl.flags import FLAGS
import numpy as np
from models import YoloV3, YoloV3Tiny
from utils import load_darknet_weights
import tensorflow as tf

flags.DEFINE_string('weights', '/data/Models/spc-yolov4/yolov3_final.weights', 'path to weights file')
flags.DEFINE_string('output', '/data/Models/spc-yolov4/convert/spc_yolo.tf', 'path to output')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('num_classes', 3, 'number of classes in the model')


def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(classes=FLAGS.num_classes)
    yolo.summary()
    logging.info('model created')

    load_darknet_weights(yolo, FLAGS.weights, FLAGS.tiny)
    logging.info('weights loaded')

    # run_model = tf.function(lambda x : yolo(x))
    # concrete_func = run_model.get_concrete_function(tf.TensorSpec([1, 416, 416, 3], tf.int8))
    img = np.random.random((1, 416, 416, 3)).astype(np.float32)
    output = yolo(img)
    logging.info('sanity check passed')

    yolo.save_weights(FLAGS.output)
    # yolo.save("/data/Models/spc-yolov4/convert/saved_model", overwrite=True, save_format="tf", signatures=concrete_func)
    logging.info('weights saved')


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
