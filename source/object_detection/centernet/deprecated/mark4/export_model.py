import tensorflow as tf
from model import CenterNet


def load_model(ckpt):
    base_model = CenterNet(inputs=input_shape, num_classes=len(classes), max_detections=max_detections, backbone=backbone)
    base_model.built = True
    base_model.load_weights(ckpt, by_name=True, skip_mismatch=True)

    print("model loaded + ckpt")
    return base_model


def single_image_infer(model, image_path):
    image = tf.io.read_file(image_path)
    image = tf.io.decode_jpeg(image, channels=3)
    resized_image = tf.image.resize(image, (input_shape[0], input_shape[1]))
    expanded_image = tf.expand_dims(resized_image, axis=0)

    prediction = model(expanded_image)[3]
    print(prediction)
    

if __name__ == "__main__":
    classes = ["face"]
    backbone = "resnet18"
    max_detections = 10
    input_shape = (512, 512, 3)
    save_path = "/data/Models/FACE_DETECTION/CenterNet"
    ckpt_path = f"{save_path}/ckpt.h5"
    img_path = "./samples/sample01.jpg"

    model = load_model(ckpt_path)
    model.summary()

    single_image_infer(model, img_path)