import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import advisor
import pandas as pd
import tensorflow as tf
from model import DeepLabV3Plus
from tflite_support.metadata_writers import image_segmenter, writer_utils

if __name__ == "__main__":
    CKPT_PATH = "/data/Models/segmentation/VOC2012-ResNet50-AUGMENT_10/best.ckpt"
    LABEL_PATH = "/data/Datasets/VOCdevkit/VOC2012/Labels/class_labels.txt"
    SAVE_PATH = f"{'/'.join(CKPT_PATH.split('/')[:-1])}/saved_model"
    TFLITE_NAME = "unity-test"
    
    IMG_SIZE = 320
    BACKBONE_NAME = CKPT_PATH.split('/')[-2].split('-')[1]
    BACKBONE_TRAINABLE = False
    FINAL_ACTIVATION =  "softmax"
    
    label_df = pd.read_csv(LABEL_PATH, lineterminator='\n', header=None, index_col=False)
    CLASSES = label_df[0].to_list()
    print(CLASSES)

    dice_loss = advisor.losses.DiceLoss()
    categorical_focal_loss = advisor.losses.CategoricalFocalLoss()
    loss = dice_loss + (1 * categorical_focal_loss)
    metrics = tf.keras.metrics.OneHotMeanIoU(num_classes=len(CLASSES))
    optimizer = tf.keras.optimizers.Adam()

    trained_model = DeepLabV3Plus(IMG_SIZE, IMG_SIZE, len(CLASSES), backbone_name=BACKBONE_NAME, backbone_trainable=BACKBONE_TRAINABLE, final_activation=FINAL_ACTIVATION)
    trained_model.load_weights(CKPT_PATH)

    squeeze = tf.keras.layers.Lambda(lambda x : tf.squeeze(x, axis=0))(trained_model.output)
    argmax = tf.keras.layers.Lambda(lambda x : tf.argmax(x, axis=-1))(squeeze)
    model = tf.keras.Model(inputs=trained_model.input, outputs=argmax)
    model.compile(optimizer=optimizer, loss=loss, metrics=[metrics])
    model.summary()

    run_model = tf.function(lambda x : model(x))
    concrete_func = run_model.get_concrete_function(tf.TensorSpec([1, IMG_SIZE, IMG_SIZE, 3], model.inputs[0].dtype))
    model.save(SAVE_PATH, overwrite=True, save_format="tf", signatures=concrete_func)

    converter = tf.lite.TFLiteConverter.from_saved_model(SAVE_PATH)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    tflite_model = converter.convert()

    with tf.io.gfile.GFile(f"{SAVE_PATH}/{TFLITE_NAME}.tflite", "wb") as f:
        f.write(tflite_model)

    ImageSegmenterWriter = image_segmenter.MetadataWriter
    _INPUT_NORM_MEAN = 127.5
    _INPUT_NORM_STD = 127.5

    writer = ImageSegmenterWriter.create_for_inference(writer_utils.load_file(f"{SAVE_PATH}/{TFLITE_NAME}.tflite"),
                                                       [_INPUT_NORM_MEAN], [_INPUT_NORM_STD], [LABEL_PATH])
    writer_utils.save_file(writer.populate(), f"{SAVE_PATH}/{TFLITE_NAME}-meta.tflite")