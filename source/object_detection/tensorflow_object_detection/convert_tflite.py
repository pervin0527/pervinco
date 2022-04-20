import tensorflow as tf
from glob import glob
from tflite_support.metadata_writers import object_detector
from tflite_support.metadata_writers import writer_utils

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

def preprocess_image(images):
    image = tf.io.read_file(images)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [640, 640])
    # image = tf.keras.applications.mobilenet.preprocess_input(image)

    return image

def representative_data_gen():
    images = sorted(glob("/data/Datasets/COCO2017/images/*"))
    idx = 0
    for input_value in tf.data.Dataset.from_tensor_slices(images).map(preprocess_image).batch(1).take(100):
        idx += 1
        print(idx)
        yield [input_value]

if __name__ == "__main__":
    int8 = True
    input_norm_mean, input_norm_std = [127.5], [127.5]
    label_file_paths='/home/barcelona/tensorflow/examples/lite/examples/object_detection/android/app/src/main/assets/labelmap.txt'
    # graph_path = "/home/barcelona/tensorflow/models/research/object_detection/jun/efficientdet_d0_coco17_tpu-32/exported_gp"
    graph_path = "/home/barcelona/tensorflow/models/research/object_detection/jun/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/export"
    saved_model_dir = f"{graph_path}/saved_model"

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    if int8:
        converter.representative_dataset = representative_data_gen
        # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.SELECT_TF_OPS]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8

    else:
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]                                       
    
    tflite_model = converter.convert()

    with tf.io.gfile.GFile(f'{graph_path}/custom.tflite', 'wb') as f:
        f.write(tflite_model)

    writer = object_detector.MetadataWriter.create_for_inference(writer_utils.load_file(f'{graph_path}/custom.tflite'), input_norm_mean=input_norm_mean, input_norm_std=input_norm_std, label_file_paths=[label_file_paths])
    writer_utils.save_file(writer.populate(), f'{graph_path}/custom_meta.tflite')