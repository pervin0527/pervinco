import cv2
import numpy as np
import tensorflow as tf
import tflite_runtime.interpreter as tflite

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


if __name__ == "__main__":
    tflite_path = "/data/Models/facial_landmark_68pts/pfld.tflite"
    classes = ["face"]
    image_path = "/data/test_image/sample02.png"

    interpreter = tflite.Interpreter(model_path = tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0].get("shape")
    input_width, input_height = input_shape[1], input_shape[2]

    image = cv2.imread(image_path)
    image = cv2.resize(image, (input_height, input_width))
    input_tensor = image / 255.0
    input_tensor = np.expand_dims(input_tensor, axis=0)
    
    interpreter.set_tensor(input_details[0]['index'], input_tensor.astype(np.float32))
    interpreter.invoke()

    landmarks = interpreter.get_tensor(output_details[0]['index'])
    landmarks = landmarks.reshape(-1, 2)
    landmarks = landmarks * [input_height, input_width]
    print(landmarks.shape)

    for (x, y) in landmarks:
        cv2.circle(image, (int(x), int (y)), radius=1, thickness=-1, color=(0, 0, 255))

    cv2.imshow("result", image)
    cv2.waitKey(0)