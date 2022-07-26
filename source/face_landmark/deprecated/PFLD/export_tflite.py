import os
import tensorflow as tf
from model import PFLD

def load_model(ckpt_path):
    model = PFLD()
    model.built = True
    model.load_weights(ckpt_path, by_name=True, skip_mismatch=True)

    return model


if __name__ == "__main__":
    input_shape = [112, 112, 3]
    save_path = "/data/Models/facial_landmark_68pts"
    ckpt_path = f"{save_path}/pfld.h5"
    
    model = load_model(ckpt_path)
    model.build = True
    model.summary()

    ### SAVED_MODEL
    run_model = tf.function(lambda x : model(x))
    concrete_func = run_model.get_concrete_function(tf.TensorSpec([1, input_shape[0], input_shape[1], 3], tf.float32))
    model.save(f"{save_path}/saved_model", overwrite=True, save_format="tf", signatures=concrete_func)
    print("Export SAVED_MODEL Done!!!")


    ### TFLITE
    converter = tf.lite.TFLiteConverter.from_saved_model(f"{save_path}/saved_model")
    converter.optimizations = [] ## [tf.lite.Optimize.DEFAULT]
    converter.experimental_new_converter = True
    converter.allow_custom_ops = False
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    tflite_model = converter.convert()

    with tf.io.gfile.GFile(f"{save_path}/saved_model/pfld.tflite", "wb") as f:
        f.write(tflite_model)

    print("Export TFLITE Done!!!")