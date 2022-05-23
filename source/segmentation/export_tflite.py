import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import yaml
import tensorflow as tf
from model import DeepLabV3Plus
from metrics import Sparse_MeanIoU
from tflite_support.metadata_writers import image_segmenter, writer_utils


def load_model_with_ckpt(ckpt_path, include_infer=False):
    if config["ONE_HOT"]:
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        metrics = tf.keras.metrics.OneHotMeanIoU(num_classes=len(config["CLASSES"]))
    else:
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metrics = Sparse_MeanIoU(num_classes=len(config["CLASSES"]))
    optimizer = tf.keras.optimizers.Adam(learning_rate=config["LR_START"])

    model = DeepLabV3Plus(IMG_SIZE, IMG_SIZE, len(CLASSES),
                          backbone_name=BACKBONE_NAME, backbone_trainable=BACKBONE_TRAINABLE,
                          final_activation=FINAL_ACTIVATION)
    model.load_weights(ckpt_path)

    if include_infer:
        inference = tf.keras.layers.Lambda(lambda x : tf.cast(tf.argmax(tf.squeeze(x, axis=0), axis=-1), dtype=tf.float32))(model.output)
        # inference = tf.keras.layers.Activation("relu")(model.output)
        model = tf.keras.Model(inputs=model.input, outputs=inference)

    model.compile(optimizer=optimizer, loss=loss, metrics=[metrics])
    return model


if __name__ == "__main__":
    model_dir = "/data/Models/segmentation/VOC2012-AUGMENT_50-ResNet101"
    label_dir = "/data/Datasets/VOCdevkit/VOC2012/Labels/labels.txt"
    ckpt = f"{model_dir}/best.ckpt"
    
    with open(f"{model_dir}/config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    SAVE_PATH = f"{model_dir}/saved_model"
    IMG_SIZE = config["IMG_SIZE"]
    BACKBONE_NAME = config["BACKBONE_NAME"]
    BACKBONE_TRAINABLE = False
    FINAL_ACTIVATION =  config["FINAL_ACTIVATION"]
    TFLITE_NAME = f"{ckpt.split('/')[-2]}"
    INCLUDE_INFER = False
    CLASSES = config["CLASSES"]
    print(CLASSES)

    model = load_model_with_ckpt(ckpt, INCLUDE_INFER)
    model.summary()

    run_model = tf.function(lambda x : model(x))
    concrete_func = run_model.get_concrete_function(tf.TensorSpec([1, IMG_SIZE, IMG_SIZE, 3], model.inputs[0].dtype))
    model.save(SAVE_PATH, overwrite=True, save_format="tf", signatures=concrete_func)
    print("Export Saved model Finished")

    converter = tf.lite.TFLiteConverter.from_saved_model(SAVE_PATH)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # converter.target_spec.supported_types = [tf.float16]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    tflite_model = converter.convert()

    with tf.io.gfile.GFile(f"{SAVE_PATH}/{TFLITE_NAME}.tflite", "wb") as f:
        f.write(tflite_model)
    print("Export tflite Finished")

    ImageSegmenterWriter = image_segmenter.MetadataWriter
    _INPUT_NORM_MEAN = 127.5
    _INPUT_NORM_STD = 127.5

    writer = ImageSegmenterWriter.create_for_inference(writer_utils.load_file(f"{SAVE_PATH}/{TFLITE_NAME}.tflite"),
                                                      [_INPUT_NORM_MEAN], [_INPUT_NORM_STD], [label_dir])
    writer_utils.save_file(writer.populate(), f"{SAVE_PATH}/{TFLITE_NAME}-meta.tflite")
    print("Export tflite with metadata Finished")