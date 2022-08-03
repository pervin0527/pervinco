# import tensorflow as tf
# from model import CenterNet, decode


# def build_pred_model(model, image_path):
#     image = tf.io.read_file(image_path)
#     image = tf.io.decode_jpeg(image, channels=3)
#     resized_image = tf.image.resize(image, (input_shape[0], input_shape[1]))
#     expanded_image = tf.expand_dims(resized_image, axis=0)

#     detections = tf.keras.layers.Lambda(lambda x: decode(*x, max_detections=max_detections))(model(expanded_image))
#     pred_model = tf.keras.Model(inputs=model.input, outputs=detections)

#     return pred_model


# def load_model(ckpt):
#     base_model = CenterNet(inputs=input_shape, num_classes=len(classes), backbone=backbone)
#     base_model.built = True
#     base_model.load_weights(ckpt, by_name=True, skip_mismatch=True)

#     print("model loaded + ckpt")
#     return base_model


# if __name__ == "__main__":
#     classes = ["face"]
#     backbone = "resnet18"
#     max_detections = 10
#     input_shape = (512, 512, 3)
#     save_path = "/data/Models/FACE_DETECTION/CenterNet"
#     ckpt_path = f"{save_path}/ckpt.h5"
#     img_path = "./samples/sample01.jpg"

#     model = load_model(ckpt_path)
#     build_pred_model(model, img_path)