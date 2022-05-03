import cv2
import numpy as np
import tensorflow as tf

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
    model_path = "/data/Models/segmentation/saved_model/model_meta.tflite"
    # model_path = "/data/Models/segmentation/lite-model_deeplabv3_1_metadata_2.tflite"
    img_size = 512
    image_path = "./dog.jpg"

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(input_details)
    print()
    print(output_details)

    image = cv2.imread(image_path)
    image = cv2.resize(image, (img_size, img_size))
    input_tensor = np.expand_dims(image, axis=0)

    # interpreter.set_tensor(input_details[0]['index'], input_tensor.astype(np.uint8))
    interpreter.set_tensor(input_details[0]['index'], input_tensor.astype(np.float32))
    interpreter.invoke()

    prediticons = interpreter.get_tensor(output_details[0]['index'])
    print(prediticons.shape)
    print(prediticons)