import os
import yaml
import tensorflow as tf
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
    model.load_weights(weight_path)
    model.summary()

    return model


if __name__ == "__main__":
    config_path = "./configs/train.yaml"
    weight_path = "/home/ubuntu/Models/centernet/unfreeze.h5"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    class_names = read_label_file(config["path"]["label_path"])
    num_classes = len(class_names)
    model = build_model()

    run_model = tf.function(lambda x : model(x))
    concrete_func = run_model.get_concrete_function(tf.TensorSpec([1, config["train"]["input_shape"][0], config["train"]["input_shape"][1], 3], model.inputs[0].dtype))
    tf.saved_model.save(model, "/home/ubuntu/Models/centernet/saved_model", signatures=concrete_func)
    print("Model SAVED")

    converter = tf.lite.TFLiteConverter.from_saved_model("/home/ubuntu/Models/centernet/saved_model")
    print("Model LOADED")

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    tflite_model = converter.convert()

    with open("/home/ubuntu/Models/centernet/CenterNet.tflite", "wb") as f:
        f.write(tflite_model)
    print("Model CONVERTED")