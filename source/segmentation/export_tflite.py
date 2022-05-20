import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
import tensorflow as tf
from model import DeepLabV3Plus
from metrics import Sparse_MeanIoU
from hparams_config import send_params
from tflite_support.metadata_writers import image_segmenter, writer_utils


def load_model_with_ckpt(ckpt_path, include_infer=False):
    if params["ONE_HOT"]:
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        metrics = tf.keras.metrics.OneHotMeanIoU(num_classes=len(params["CLASSES"]))
    else:
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metrics = Sparse_MeanIoU(num_classes=len(params["CLASSES"]))
    optimizer = tf.keras.optimizers.Adam(learning_rate=params["LR_START"])

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
    CKPT_PATH = "/data/Models/segmentation/TEST-BASIC/best.ckpt"

    params = send_params(show_contents=False)
    SAVE_PATH = params["SAVE_PATH"] + "/saved_model"
    IMG_SIZE = params["IMG_SIZE"]
    BACKBONE_NAME = params["BACKBONE_NAME"]
    BACKBONE_TRAINABLE = False
    FINAL_ACTIVATION =  params["FINAL_ACTIVATION"]
    TFLITE_NAME = f"{CKPT_PATH.split('/')[-2]}"
    INCLUDE_INFER = False
    
    CLASSES = params["CLASSES"]
    print(CLASSES)

    model = load_model_with_ckpt(CKPT_PATH, INCLUDE_INFER)
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
                                                      [_INPUT_NORM_MEAN], [_INPUT_NORM_STD], [params["LABEL_PATH"]])
    writer_utils.save_file(writer.populate(), f"{SAVE_PATH}/{TFLITE_NAME}-meta.tflite")
    print("Export tflite with metadata Finished")