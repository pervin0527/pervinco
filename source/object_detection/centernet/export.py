import os
import yaml
import tensorflow as tf
from glob import glob
from models.centernet import centernet
from data.data_utils import read_label_file

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
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


def build_model():
    model = centernet([config["train"]["input_shape"][0], config["train"]["input_shape"][1], 3],
                      num_classes,
                      backbone=config["train"]["backbone"],
                      max_objects=config["train"]["max_detection"],
                      mode="predict")
    model.load_weights(config["export"]["ckpt_path"])
    model.summary()

    return model


def preprocess_image(images):
    image = tf.io.read_file(images)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [512, 512])
    image = image / 127.5 - 1
    # image = image / 255.

    return image

def representative_data_gen():
    data_path = config["export"]["representative_data_path"]
    images = sorted(glob(data_path + "/*"))
    idx = 0
    for input_value in tf.data.Dataset.from_tensor_slices(images).map(preprocess_image).batch(1).take(100):
        idx += 1
        
        yield [input_value]


def detail_viewer(detail):
    for info in detail:
        print(info)
        print()


if __name__ == "__main__":
    with open("./configs.yaml", "r") as f:
        config = yaml.safe_load(f)

    class_names = read_label_file(config["path"]["label_path"])
    num_classes = len(class_names)
    model = build_model()

    run_model = tf.function(lambda x : model(x))
    concrete_func = run_model.get_concrete_function(tf.TensorSpec([1, config["train"]["input_shape"][0], config["train"]["input_shape"][1], 3], model.inputs[0].dtype))
    tf.saved_model.save(model, config["export"]["saved_model_path"], signatures=concrete_func)
    print("Model SAVED")

    converter = tf.lite.TFLiteConverter.from_saved_model(config["export"]["saved_model_path"])
    print("Model LOADED")

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.experimental_new_converter = True
    converter.allow_custom_ops = True
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.SELECT_TF_OPS]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.float32
    tflite_model = converter.convert()

    tflite_save_path = config["path"]["save_path"] + "/" + config["export"]["tflite_file_name"]
    with open(tflite_save_path, "wb") as f:
        f.write(tflite_model)
    print("Model CONVERTED")

    interpreter = tf.lite.Interpreter(tflite_save_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    detail_viewer(input_details)
    detail_viewer(output_details)