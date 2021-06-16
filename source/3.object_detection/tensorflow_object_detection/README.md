## Guide
[Tensorflow 2 Object Detection API](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md)

## change config files
[tf2 object detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)

<p align="center"><img src="../0.doc_imgs/../../0.doc_imgs/tf2_odt_cfg.png"></p>


## training

    # cd tensorflow/models/research

    python3 object_detection/model_main_tf2.py \
    --pipeline_config_path=object_detection/custom/deploy/efficientdet/pipeline.config \
    --model_dir=object_detection/custom/models/21_06_09_efnet

## Tensorboard
    # cd tensorflow/models/research

    tensorboard --logdir=./object_detection/custom/models/21_06_09_efnet/train

## Convert pb, tflite

    python3 object_detection/export_tflite_graph_tf2.py
    --pipeline_config_path object_detection/custom/deploy/ssd_mobilenet_v2_320/pipeline.config 
    --trained_checkpoint_dir object_detection/custom/models/traffic_sign/21_06_14
    --output_directory object_detection/custom/models/traffic_sign/21_06_14/

    python3 convert_tflite.py

## EfficientDet

    python3 object_detection/export_tflite_graph_tf2.py --pipeline_config_path=object_detection/custom/deploy/efficientdet/pipeline.config --trained_checkpoint_dir=object_detection/custom/models/traffic_sign/21_06_15 --output_directory=object_detection/custom/models/traffic_sign/21_06_15/ --config_override="\
    model{ \
      ssd{ \
        image_resizer { \
          fixed_shape_resizer { \
            height: 512 \
            width: 512 \
          } \
        } \
      } \
    }"

