import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import numpy as np
import tensorflow as tf
from glob import glob

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
    with strategy.scope():
        base_model = tf.keras.applications.ResNet50(input_shape=(img_size, img_size, 3), include_top=False, weights="imagenet")    
        base_model.trainable = True

        input_layer = tf.keras.Input(shape=(img_size, img_size, 3))
        x = tf.keras.applications.resnet50.preprocess_input(input_layer)
        x = base_model(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.SeparableConv2D(keypoints, kernel_size=5, strides=1, activation="relu")(x)
        output_layer = tf.keras.layers.SeparableConv2D(keypoints, kernel_size=3, strides=1, activation="sigmoid")(x)

        model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
        model.summary()

        return model

def inference_images(dir):
    if not os.path.isdir(f"{dir}/inference"):
        os.makedirs(f"{dir}/inference")

    test_dir = sorted(glob(f"{dir}/*.jpg"))
    for idx, test_file in enumerate(test_dir):
        image = cv2.imread(test_file)
        result_image = image.copy()
        image = cv2.resize(image, (img_size, img_size))

        predictions = model.predict(np.expand_dims(image, axis=0), verbose=0)
        predictions = predictions.reshape(-1, 98, 2) * img_size

        for (x, y) in predictions[0]:
            cv2.circle(result_image, (int(x), int(y)), radius=1, color=(0, 0, 255), thickness=3)

        cv2.imwrite(f"./inference/{idx}.jpg", result_image)


def inference_video(viz_height, viz_width):
    capture = cv2.VideoCapture(-1)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while cv2.waitKey(33) != ord('q'):
        ret, frame = capture.read()
        
        image = cv2.resize(frame, (img_size, img_size))
        result_image = image.copy()
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        input_tensor = np.expand_dims(image, axis=0)

        prediction = model.predict(input_tensor)
        prediction = prediction.reshape(-1, 98, 2) * img_size

        for (x, y) in prediction[0]:
            cv2.circle(result_image, (int(x), int(y)), radius=1, color=(0, 0, 255), thickness=3)            

        result_image = cv2.resize(result_image, (viz_height, viz_width))
        cv2.imshow("result", result_image)
        
    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    img_size = 224
    keypoints = 98 * 2
    inference_type = "video"
    test_images = "/data/Datasets/WFLW/test/images"

    model = build_model()
    model.load_weights("/data/Models/facial_landmark/best.ckpt")
    print("model ckpt loaded")

    if inference_type == "images":
        inference_images(test_images)

    elif inference_type == "video":
        inference_video(512, 512)