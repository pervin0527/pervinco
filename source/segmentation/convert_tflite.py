import tensorflow as tf
from glob import glob
from tflite_support.metadata_writers import image_segmenter, writer_utils

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
    image = tf.image.resize(image, [image_size, image_size])

    return image

def representative_data_gen():
    images = sorted(glob(f"{representative_data_path}/*"))
    idx = 0
    for input_value in tf.data.Dataset.from_tensor_slices(images).map(preprocess_image).batch(1).take(100):
        idx += 1
        
        yield [input_value]

if __name__ == "__main__":
    int8 = False
    image_size = 512
    saved_model_path = "/data/Models/segmentation"
    folder = "custom-softmax"
    label_file_path = "/data/Datasets/VOCdevkit/VOC2012/Labels/class_labels.txt"
    representative_data_path = "/data/Datasets/VOCdevkit/VOC2012/Segmentation/valid/images"

    lite_name = None
    if not int8:
        lite_name = "fp32"
    else:
        lite_name = "int8"

    converter = tf.lite.TFLiteConverter.from_saved_model(f"{saved_model_path}/{folder}/saved_model")
    # model = tf.keras.models.load_model(saved_model_path)
    # model.input.set_shape((1, 512, 512, 3))
    # converter = tf.lite.TFLiteConverter.from_saved_model(model)
    
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    if int8:
        converter.representative_dataset = representative_data_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8

    else:
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

    tflite_model = converter.convert()

    with tf.io.gfile.GFile(f"{saved_model_path}/{folder}/{folder}_{lite_name}.tflite", "wb") as f:
        f.write(tflite_model)

    print("saved model converted tflite")

    ImageSegmenterWriter = image_segmenter.MetadataWriter
    _INPUT_NORM_MEAN = 127.5
    _INPUT_NORM_STD = 127.5

    writer = ImageSegmenterWriter.create_for_inference(writer_utils.load_file(f"{saved_model_path}/{folder}/{folder}_{lite_name}.tflite"), [_INPUT_NORM_MEAN], [_INPUT_NORM_STD], [label_file_path])

    print(writer.get_metadata_json())
    writer_utils.save_file(writer.populate(), f"{saved_model_path}/{folder}/{folder}_m{lite_name}.tflite")

    print("tflite with metadata")