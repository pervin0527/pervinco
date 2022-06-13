import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import numpy as np
import tensorflow as tf
from glob import glob

def build_model():
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

if __name__ == "__main__":
    img_size = 224
    keypoints = 98 * 2

    model = build_model()
    model.load_weights("/home/ubuntu/Models/facial_landmark/best.ckpt")
    print("model ckpt loaded")

    if not os.path.isdir("./inference"):
        os.makedirs("./inference")

    test_dir = sorted(glob("/home/ubuntu/Datasets/WFLW/test/images/*"))
    for idx, test_file in enumerate(test_dir[:10]):
        image = cv2.imread(test_file)
        result_image = image.copy()
        image = cv2.resize(image, (img_size, img_size))

        predictions = model.predict(np.expand_dims(image, axis=0), verbose=0)
        predictions = predictions.reshape(-1, 98, 2) * img_size

        for (x, y) in predictions[0]:
            cv2.circle(result_image, (int(x), int(y)), radius=1, color=(0, 0, 255), thickness=3)

        cv2.imwrite(f"./inference/{idx}.jpg", result_image)