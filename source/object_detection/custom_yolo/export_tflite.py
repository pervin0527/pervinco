import cv2
import numpy as np
import tensorflow as tf
from glob import glob
from models import YoloV3, YoloV3Tiny

def preprocessing(images):
    image = tf.io.read_file(images)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [416, 416])
    image = tf.cast(image, tf.uint8)

    return image

def representative_data_gen():
    images = glob('/data/Datasets/SPC/full-name14/valid3/images/*.jpg')
    for input_value in tf.data.Dataset.from_tensor_slices(images).map(preprocessing).batch(1).take(100):
        yield [input_value]
   
if __name__ == '__main__':
    yolo = YoloV3(size=416, classes=3, max_detections=1, score_threshold=0.5, iou_threshold=0.4)
    yolo.load_weights('/data/Models/spc-yolov4/convert/spc_yolo.tf')

    input_layer = tf.keras.Input([416, 416, 3], dtype=tf.uint8)
    dtype_layer = tf.keras.layers.Lambda(lambda x : tf.cast(x, tf.float32) / 255.)(input_layer)
    output_layer = yolo(dtype_layer)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    model.summary()

    run_model = tf.function(lambda x : model(x))
    concrete_func = run_model.get_concrete_function(tf.TensorSpec([1, 416, 416, 3], model.inputs[0].dtype))
    tf.saved_model.save(model, "/data/Models/spc-yolov4/convert/saved_model", signatures=concrete_func)

    converter = tf.lite.TFLiteConverter.from_saved_model("/data/Models/spc-yolov4/convert/saved_model")
    # converter.experimental_new_converter = True
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    # converter.representative_dataset = representative_data_gen
    # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # converter.inference_input_type = tf.uint8
    # converter.inference_output_type = tf.float32
    tflite_model = converter.convert()

    with open("/data/Models/spc-yolov4/convert/custom_yolo.tflite", "wb") as f:
        f.write(tflite_model)

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
    
    scores = interpreter.get_tensor(output_details[0]['index'])
    valid_detections = interpreter.get_tensor(output_details[1]['index'])
    bboxes = interpreter.get_tensor(output_details[2]['index'])
    classes = interpreter.get_tensor(output_details[3]['index'])

    print(scores.shape, bboxes.shape, classes.shape)
    class_names = [c.strip() for c in open("/data/Datasets/SPC/Labels/labels.txt").readlines()]

    result_image = image.numpy()
    indices = np.where(scores[0] > 0.6)
    for idx in indices:
        bbox = bboxes[0][idx]
        score = scores[0][idx]
        label = classes[0][idx]
        print(bbox, score, label)

        xmin, ymin, xmax, ymax = int(bbox[0][0] * 416), int(bbox[0][1] * 416), int(bbox[0][2] * 416), int(bbox[0][3] * 416)
        cv2.rectangle(result_image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
        print(class_names[int(label[0])], score[0])

    cv2.imwrite("tflite_result.jpg", result_image)
