import os
import cv2
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
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


@tf.function
def detect_fn(image, detection_model):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)

    return detections, prediction_dict, tf.reshape(shapes, [-1])


if __name__ == "__main__":
    PATH_TO_CFG = '/home/barcelona/tensorflow/models/research/object_detection/custom/deploy/ssd_mobilenet_v2_320/pipeline.config'
    PATH_TO_CKPT = '/home/barcelona/tensorflow/models/research/object_detection/custom/models/traffic_sign/21_06_14'
    PATH_TO_LABELS = '/home/barcelona/tensorflow/models/research/object_detection/custom/labels/traffic_sign.txt'
    CKPT_VALUE = 'ckpt-101'
    THRESH_HOLD = .7

    ###############################################################################################
    cap = cv2.VideoCapture(-1)
    MJPG_CODEC = 1196444237.0 # MJPG
    cap_AUTOFOCUS = 0
    cap_FOCUS = 0
    #cap_ZOOM = 400

    frame_width = int(1920)
    frame_height = int(1080)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    # cv2.namedWindow('inference', cv2.WINDOW_FREERATIO)
    # cv2.resizeWindow('inference', frame_width, frame_height)

    cap.set(cv2.CAP_PROP_BRIGHTNESS, 0)
    cap.set(cv2.CAP_PROP_FOURCC, MJPG_CODEC)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, cap_AUTOFOCUS)
    cap.set(cv2.CAP_PROP_FOCUS, cap_FOCUS)
    ##############################################################################################

    configs = config_util.get_configs_from_pipeline_file(PATH_TO_CFG)
    model_config = configs['model']
    detection_model = model_builder.build(model_config=model_config, is_training=False)

    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(os.path.join(PATH_TO_CKPT, CKPT_VALUE)).expect_partial()

    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

    while True:
        ret, image_np = cap.read()
        print(image_np.shape)

        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections, predictions_dict, shapes = detect_fn(input_tensor, detection_model)

        label_id_offset = 1
        image_np_with_detections = image_np.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(image_np_with_detections,
                                                            detections['detection_boxes'][0].numpy(),
                                                            (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
                                                            detections['detection_scores'][0].numpy(),
                                                            category_index,
                                                            use_normalized_coordinates=True,
                                                            max_boxes_to_draw=100,
                                                            min_score_thresh=THRESH_HOLD,
                                                            agnostic_mode=False)

        # Display output
        cv2.imshow('object detection', cv2.resize(image_np_with_detections, (640, 480)))

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()