# Guide
[Tensorflow 2 Object Detection API](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md)

NOTE: TFLite currently only supports SSD Architectures (excluding EfficientDet) for boxes-based detection. Support for EfficientDet is provided via the TFLite Model Maker library.

The output model has the following inputs & outputs:


## change config files
[tf2 object detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)

<p align="left"><img src="./../../../doc_imgs/tf2_odt_cfg.png" width=85%></p>


## training

    # cd tensorflow/models/research
    python3 model_main_tf2.py \
    --alsologtostderr \
    --model_dir=$out_dir \ 
    --checkpoint_every_n=500  \
    --pipeline_config_path=../models/ssd_mobilenet_v2_raccoon.config \
    --eval_on_train_data 2>&1 | tee $out_dir/train.log

    python3 model_main_tf2.py 
    --alsologtostderr --model_dir=$out_dir \
    --pipeline_config_path=../models/ssd_mobilenet_v2_raccoon.config \
    --checkpoint_dir=$out_dir  2>&1 | tee $out_dir/eval.log


## Tensorboard
  if you train on server and want to monitor tensorboard.

  oooo : tensorboard port number  
  xxxx : ip address  
  ^^^^ : ip port


    ssh -L oooo:localhost:oooo -p ^^^^ name@xxxx.xxxx.xxxx.xxxx

    # shutdown process
    lsof -i:6006
    kill -9 PID  

    # cd tensorflow/models/research

    tensorboard --logdir=./object_detection/custom/models/21_06_09_efnet/train

## Convert checkpoint to SavedModel

    # cd tensorflow/models/research

    python3 object_detection/exporter_main_v2.py \
    --input_type=float_image_tensor \
    --pipeline_config_path=object_detection/custom/deploy/ssd_mobilenet_v2_320/pipeline.config \
    --trained_checkpoint_dir=object_detection/custom/models/fire/21_07_08 \
    --output_directory=/home/barcelona/test/ssd_mb_v2/

    # optional
    python3 -m tf2onnx.convert --saved-model saved_model/ --opset 11 --output ./model.onnx --inputs-as-nchw input_tensor:0

## Convert pb, tflite

    python3 object_detection/export_tflite_graph_tf2.py
    --pipeline_config_path object_detection/custom/deploy/ssd_mobilenet_v2_320/pipeline.config 
    --trained_checkpoint_dir object_detection/custom/models/traffic_sign/21_06_14
    --output_directory object_detection/custom/models/traffic_sign/21_06_14/

    python3 convert_tflite.py

## EfficientDet

    python3 object_detection/export_tflite_graph_tf2.py \
    --pipeline_config_path object_detection/custom/deploy/efficientdet/pipeline.config \
    --trained_checkpoint_dir object_detection/custom/models/traffic_sign/21_06_17 \
    --output_directory object_detection/custom/models/traffic_sign/21_06_17 \
    --config_override="\
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

## CenterNet
  if you use centernet, refer to this link  
  [https://github.com/tensorflow/models/issues/9414#issuecomment-791674050](https://github.com/tensorflow/models/issues/9414#issuecomment-791674050)

    python3 object_detection/export_tflite_graph_tf2.py \
    --pipeline_config_path object_detection/custom/deploy/centernet_resnet50_v2_512/pipeline.config \
    --trained_checkpoint_dir object_detection/custom/models/traffic_sign/21_06_18 \
    --output_directory object_detection/custom/models/traffic_sign/21_06_18 \
    --centernet_include_keypoints=false \
    --max_detections=10 \
    --config_override="\
    model{ \
      center_net{ \
        image_resizer { \
          fixed_shape_resizer { \
            height: 512 \
            width: 512 \
          } \
        } \
      } \
    }"

## EfficientDet lite with Model maker
[https://www.tensorflow.org/lite/tutorials/model_maker_object_detection](https://www.tensorflow.org/lite/tutorials/model_maker_object_detection)
[https://www.tensorflow.org/lite/api_docs/python/tflite_model_maker/config/QuantizationConfig#for_int8](https://www.tensorflow.org/lite/api_docs/python/tflite_model_maker/config/QuantizationConfig#for_int8)

    python3 model_maker.py

## convert EfficientDet lite to ONNX

    python3 -m tf2onnx.convert --opset 13 --tflite fire_efdet_d0.tflite --output fire_efdet_d0.onnx


## tensorflow js

    # model converter
    tensorflowjs_converter \ 
    --input_format=tf_saved_model \
    /data/Models/test/model/saved_model/ \
    /data/Models/js

