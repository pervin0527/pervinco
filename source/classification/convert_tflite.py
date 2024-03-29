import numpy as np
import tensorflow as tf
from glob import glob
from sklearn import preprocessing
from tflite_support.metadata_writers import writer_utils
from tflite_support.metadata_writers import image_classifier
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet import preprocess_input

def preprocess_image(images):
    image = tf.io.read_file(images)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = tf.keras.applications.mobilenet.preprocess_input(image)

    return image

def representative_data_gen():
    images = sorted(glob("/data/Datasets/SPC/Cls/test/images/*"))
    for input_value in tf.data.Dataset.from_tensor_slices(images).map(preprocess_image).batch(1).take(100):
        yield [input_value]


saved_model_dir = "/data/Models/classification/SPC/2022.03.24_18:18"
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# converter.target_spec.supported_types = []
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
# converter.experimental_new_quantizer = False

tflite_model = converter.convert()

with open(f"{saved_model_dir}/test.tflite", "wb") as f:
    f.write(tflite_model)

ImageClassifierWriter = image_classifier.MetadataWriter
MODEL_PATH = f"{saved_model_dir}/test.tflite"
LABEL_FILE = "/data/Datasets/SPC/Labels/labels.txt"
SAVE_TO_PATH = f"{saved_model_dir}/test_metadata.tflite"

INPUT_NORM_MEAN = 127.5
INPUT_NORM_STD = 127.5

# Create the metadata writer.
writer = ImageClassifierWriter.create_for_inference(
    writer_utils.load_file(MODEL_PATH), [INPUT_NORM_MEAN], [INPUT_NORM_STD], [LABEL_FILE])

# Verify the metadata generated by metadata writer.
print(writer.get_metadata_json())

# Populate the metadata into the model.
writer_utils.save_file(writer.populate(), SAVE_TO_PATH)