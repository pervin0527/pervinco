import os
import cv2
import numpy as np
import tensorflow as tf
from glob import glob
from models import YoloV3

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

def preprocessing(images):
    image = tf.io.read_file(images)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [416, 416])
    image = image / 255.
    # image = tf.cast(image, tf.uint8)

    return image

def representative_data_gen():
    images = glob('/data/Datasets/SPC/full-name14/valid3/images/*.jpg')
    for input_value in tf.data.Dataset.from_tensor_slices(images).map(preprocessing).batch(1).take(100):
        yield [input_value]
   
if __name__ == '__main__':
    yolo = YoloV3(size=416, classes=3, max_detections=1, score_threshold=0.5, iou_threshold=0.4)
    yolo.load_weights('/data/Models/spc-yolov4/convert/spc_yolo.tf')

    # input_layer = tf.keras.Input([416, 416, 3], dtype=tf.uint8)
    # dtype_layer = tf.keras.layers.Lambda(lambda x : tf.cast(x, tf.float32) / 255.)(input_layer)
    # output_layer = yolo(dtype_layer)

    # model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    # model.summary()

    model = yolo

    run_model = tf.function(lambda x : model(x))
    concrete_func = run_model.get_concrete_function(tf.TensorSpec([1, 416, 416, 3], model.inputs[0].dtype, name="images"))
    tf.saved_model.save(model, "/data/Models/spc-yolov4/convert/saved_model", signatures=concrete_func)
    print("Model SAVED")

    converter = tf.lite.TFLiteConverter.from_saved_model("/data/Models/spc-yolov4/convert/saved_model")
    print("Model LOADED")
    converter.experimental_new_converter = True
    converter.experimental_new_quantizer = True
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.SELECT_TF_OPS]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.float32
    tflite_model = converter.convert()

    with open("/data/Models/spc-yolov4/convert/custom_yolo.tflite", "wb") as f:
        f.write(tflite_model)
    print("Model CONVERTED")

    interpreter = tf.lite.Interpreter("/data/Models/spc-yolov4/convert/custom_yolo.tflite")
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    print(input_details)
    print()
    output_details = interpreter.get_output_details()
    print(output_details)

    image = tf.image.decode_image(open("./test_imgs/1_0001.jpg", 'rb').read(), channels=3)
    image = tf.image.resize(image, (416, 416))
    input_tensor = tf.expand_dims(image, 0)

    interpreter.set_tensor(input_details[0]['index'], tf.cast(input_tensor, tf.uint8))
    interpreter.invoke()
    
    output0 = interpreter.get_tensor(output_details[0]['index']) # scores
    output1 = interpreter.get_tensor(output_details[1]['index']) # class_id
    output2 = interpreter.get_tensor(output_details[2]['index']) # bboxes
    output3 = interpreter.get_tensor(output_details[3]['index']) # valid_detections