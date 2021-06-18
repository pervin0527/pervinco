import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2

from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import config_util
from object_detection.builders import model_builder

# GPU setup
gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 1:
    try:
        print("Activate Multi GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
    except RuntimeError as e:
        print(e)

else:
    try:
        print("Activate Sigle GPU")
        tf.config.experimental.set_memory_growth(gpus[0], True)
        strategy = tf.distribute.experimental.CentralStorageStrategy()
    except RuntimeError as e:
        print(e)

def plot_detections(image_np,
                    boxes,
                    classes,
                    scores,
                    category_index,
                    keypoints=None,
                    keypoint_scores=None,
                    figsize=(12, 16),
                    image_name=None):

  keypoint_edges = [(0, 1),
        (0, 2),
        (1, 3),
        (2, 4),
        (0, 5),
        (0, 6),
        (5, 7),
        (7, 9),
        (6, 8),
        (8, 10),
        (5, 6),
        (5, 11),
        (6, 12),
        (11, 12),
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16)]
  image_np_with_annotations = image_np.copy()
  # Only visualize objects that get a score > 0.3.
  viz_utils.visualize_boxes_and_labels_on_image_array(
      image_np_with_annotations,
      boxes,
      classes,
      scores,
      category_index,
      keypoints=keypoints,
      keypoint_scores=keypoint_scores,
      keypoint_edges=keypoint_edges,
      use_normalized_coordinates=True,
      min_score_thresh=0.3)
  if image_name:
    plt.imsave(image_name, image_np_with_annotations)

  else:
    return image_np_with_annotations

def detect(interpreter, input_tensor, include_keypoint=False):
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  interpreter.set_tensor(input_details[0]['index'], input_tensor.numpy())

  interpreter.invoke()

  boxes = interpreter.get_tensor(output_details[0]['index'])
  classes = interpreter.get_tensor(output_details[1]['index'])
  scores = interpreter.get_tensor(output_details[2]['index'])
  num_detections = interpreter.get_tensor(output_details[3]['index'])

  if include_keypoint:
    kpts = interpreter.get_tensor(output_details[4]['index'])
    kpts_scores = interpreter.get_tensor(output_details[5]['index'])
    return boxes, classes, scores, num_detections, kpts, kpts_scores

  else:
    return boxes, classes, scores, num_detections

if __name__ == "__main__":
    model_path = '/home/barcelona/tensorflow/models/research/object_detection/custom/models/traffic_sign/21_06_17/custom.tflite'
    label_map_path = '/home/barcelona/tensorflow/models/research/object_detection/custom/labels/traffic_sign.txt'
    image_path = '/data/datasets/traffic_sign/test/trafficlight.jpg'

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    category_index = label_map_util.create_category_index_from_labelmap(
        label_map_path)

    label_id_offset = 1

    image = tf.io.read_file(image_path)
    image = tf.compat.v1.image.decode_jpeg(image)
    image = tf.expand_dims(image, axis=0)
    image_numpy = image.numpy()

    input_tensor = tf.convert_to_tensor(image_numpy, dtype=tf.float32)
    input_tensor = tf.image.resize(input_tensor, (512, 512))
    boxes, classes, scores, num_detections = detect(interpreter, input_tensor)

    vis_image = plot_detections(image_numpy[0],
                                boxes[0],
                                classes[0].astype(np.uint32) + label_id_offset,
                                scores[0],
                                category_index)

    vis_image = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
    cv2.imshow('result', vis_image)
    cv2.waitKey(0)