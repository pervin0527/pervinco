## change config files

## training

    # cd tensorflow/models/research

    python3 object_detection/model_main_tf2.py \
    --pipeline_config_path=object_detection/custom/deploy/efficientdet/pipeline.config \
    --model_dir=object_detection/custom/models/21_06_09_efnet

## Tensorboard
    # cd tensorflow/models/research

    tensorboard --logdir=./object_detection/custom/models/21_06_09_efnet/train