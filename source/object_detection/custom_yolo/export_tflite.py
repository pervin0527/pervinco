from absl import app, flags, logging
from absl.flags import FLAGS

import cv2
import numpy as np
import tensorflow as tf
from models import YoloV3, YoloV3Tiny
from dataset import transform_images

from glob import glob
from tensorflow.python.eager import def_function
from tensorflow.python.framework import tensor_spec
from tensorflow.python.util import nest

flags.DEFINE_string('weights', '/data/Models/spc-yolov4/convert/spc_yolo.tf', 'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_string('output', '/data/Models/spc-yolov4/convert/custom_yolo2.tflite', 'path to saved_model')
flags.DEFINE_string('classes', '/data/Datasets/SPC/Labels/labels.txt', 'path to classes file')
flags.DEFINE_string('image', './test_imgs/1_0004.jpg', 'path to input image')
flags.DEFINE_integer('num_classes', 3, 'number of classes in the model')
flags.DEFINE_integer('size', 416, 'image size')
flags.DEFINE_boolean("uint8", True, "quaintize int")

def preprocessing(images):
    image = tf.io.read_file(images)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [FLAGS.size, FLAGS.size])
    image = tf.cast(image, tf.uint8)

    return image


def representative_data_gen():
    images = glob('/data/Datasets/SPC/full-name14/valid3/images/*.jpg')
    for input_value in tf.data.Dataset.from_tensor_slices(images).map(preprocessing).batch(1).take(100):
        yield [input_value]


def main(_argv):
    if FLAGS.tiny:
        yolo = YoloV3Tiny(size=FLAGS.size, classes=FLAGS.num_classes, max_detections=1, score_threshold=0.5, iou_threshold=0.4)
    else:
        yolo = YoloV3(size=FLAGS.size, classes=FLAGS.num_classes, max_detections=1, score_threshold=0.5, iou_threshold=0.4)
    yolo.load_weights(FLAGS.weights)
    logging.info('weights loaded')

    input_layer = tf.keras.Input([FLAGS.size, FLAGS.size, 3], dtype=tf.uint8)
    dtype_layer = tf.keras.layers.Lambda(lambda x : tf.cast(x, tf.float32) / 255.)(input_layer)
    output_layer = yolo(dtype_layer)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    model.summary()

    run_model = tf.function(lambda x : model(x))
    concrete_func = run_model.get_concrete_function(tf.TensorSpec([1, FLAGS.size, FLAGS.size, 3], model.inputs[0].dtype))
    tf.saved_model.save(model, "/data/Models/spc-yolov4/convert/saved_model", signatures=concrete_func)
    logging.info("model saved")

    converter = tf.lite.TFLiteConverter.from_saved_model("/data/Models/spc-yolov4/convert/saved_model")
    logging.info("model loaded")
    converter.experimental_new_converter = True
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    # converter.representative_dataset = representative_data_gen
    # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # converter.inference_input_type = tf.uint8
    # converter.inference_output_type = tf.float32
    tflite_model = converter.convert()

    with open("/data/Models/spc-yolov4/convert/custom_yolo.tflite", "wb") as f:
        f.write(tflite_model)
    logging.info("tflite exported")

    interpreter = tf.lite.Interpreter("/data/Models/spc-yolov4/convert/custom_yolo.tflite")
    logging.info("tflite loaded")
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    print(input_details)
    print()
    output_details = interpreter.get_output_details()
    print(output_details)

    image = tf.image.decode_image(open(FLAGS.image, 'rb').read(), channels=3)
    image = tf.image.resize(image, (FLAGS.size, FLAGS.size))
    input_tensor = tf.expand_dims(image, 0)

    interpreter.set_tensor(input_details[0]['index'], tf.cast(input_tensor, tf.uint8))
    interpreter.invoke()
    
    scores = interpreter.get_tensor(output_details[0]['index'])
    valid_detections = interpreter.get_tensor(output_details[1]['index'])
    bboxes = interpreter.get_tensor(output_details[2]['index'])
    classes = interpreter.get_tensor(output_details[3]['index'])

    print(scores.shape, bboxes.shape, classes.shape)
    
    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')

    result_image = image.numpy()
    indices = np.where(scores[0] > 0.6)
    for idx in indices:
        bbox = bboxes[0][idx]
        score = scores[0][idx]
        label = classes[0][idx]
        print(bbox, score, label)

        xmin, ymin, xmax, ymax = int(bbox[0][0] * FLAGS.size), int(bbox[0][1] * FLAGS.size), int(bbox[0][2] * FLAGS.size), int(bbox[0][3] * FLAGS.size)
        cv2.rectangle(result_image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
        print(class_names[int(label[0])], score[0])

    cv2.imwrite("tflite_result.jpg", result_image)
    
    
if __name__ == '__main__':
    app.run(main)
