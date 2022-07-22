import os
import cv2
import numpy as np
import tensorflow as tf

from glob import glob
from model import centernet
from optimizer import AngularGrad
from data_generator import Datasets
from IPython.display import clear_output
from generators.pascal import PascalVocGenerator


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


def preprocess_image(image):
    image = image.astype(np.float32)

    image[..., 0] -= 103.939
    image[..., 1] -= 116.779
    image[..., 2] -= 123.68

    return image


def draw_result(idx, image, detections):
    scores = detections[:, 4]
    indices = np.where(scores > 0.7)[0]
    print(indices, detections[indices])
    detections[:, [0, 2]] = np.clip(detections[:, [0, 2]], 0, image.shape[1])
    detections[:, [1, 3]] = np.clip(detections[:, [1, 3]], 0, image.shape[0])

    if len(indices):
        result_image = image.copy()
        for result in detections[indices]:
            xmin, ymin, xmax, ymax, score, label = int(result[0]), int(result[1]), int(result[2]), int(result[3]), result[4], int(result[5])
            cv2.rectangle(result_image, (xmin, ymin), (xmax, ymax), (0, 0, 255))

        cv2.imwrite("./epoch_end/result.jpg", result_image)


def plot_predictions(model):
    for idx, file in enumerate(sorted(glob("./samples/*"))):
        image = cv2.imread(file)
        image = cv2.resize(image, (input_shape[0], input_shape[1]))
        input_tensor = preprocess_image(image)
        input_tensor = np.expand_dims(input_tensor, axis=0)

        prediction = model.predict(input_tensor, verbose=0)[0]
        draw_result(idx, image, prediction)


class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)

        if not os.path.isdir("./epoch_end"):
            os.makedirs("./epoch_end")

        plot_predictions(model=prediction_model)


if __name__ == "__main__":
    # train_data_dir = "/home/ubuntu/Datasets/WIDER/CUSTOM_XML/train"
    # test_data_dir = "/home/ubuntu/Datasets/WIDER/CUSTOM_XML/test"

    freeze_backbone = False
    backbone = "resnet101"
    classes = ["face"]
    epochs = 300
    batch_size = 64
    max_detections = 30
    input_shape = (512, 512, 3)

    if freeze_backbone:
        learning_rate = 0.001
        save_dir = "/home/ubuntu/Models/CenterNet/custom.h5"
        ckpt_path = ""

    else:
        learning_rate = 0.0001
        save_dir = "/home/ubuntu/Models/CenterNet/custom_unfreeze.h5"
        ckpt_path = "/home/ubuntu/Models/CenterNet/custom_unfreeze.h5"
    
    # train_dataset = Datasets(train_data_dir, classes, batch_size, input_shape[0], max_detections, shuffle=True)
    # test_dataset = Datasets(test_data_dir, classes, batch_size, input_shape[0], max_detections, shuffle=False)

    # train_steps = int(tf.math.ceil(train_dataset.size() / batch_size).numpy())
    # test_steps = int(tf.math.ceil(test_dataset.size() / batch_size).numpy())

    train_generator = PascalVocGenerator(
        "/home/ubuntu/Datasets/300VW_Dataset_2015_12_14/face_detection/train_512",
        'list',
        skip_difficult=True,
        skip_truncated=True,
        multi_scale=False,
        misc_effect=False,
        visual_effect=False,
        batch_size=batch_size,
        input_size=512,
    )

    test_generator = PascalVocGenerator(
        "/home/ubuntu/Datasets/300VW_Dataset_2015_12_14/face_detection/test_512",
        'list',
        skip_difficult=True,
        skip_truncated=True,
        multi_scale=False,
        misc_effect=False,
        visual_effect=False,
        batch_size=batch_size,
        input_size=512,
    )

    # optimizer = AngularGrad(method_angle="cos", learning_rate=learning_rate)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    cdr = tf.keras.optimizers.schedules.CosineDecayRestarts(initial_learning_rate=learning_rate,
                                                            first_decay_steps=100,
                                                            t_mul=2.0,
                                                            m_mul=0.3,
                                                            alpha=0.001)
    callbacks = [DisplayCallback(),
                 tf.keras.callbacks.LearningRateScheduler(cdr),
                 tf.keras.callbacks.ModelCheckpoint(save_dir, monitor="val_loss", verbose=1, save_best_only=True, save_weights_only=True)]

    with strategy.scope():
        model, prediction_model, debug_model = centernet(input_shape=input_shape, num_classes=len(classes), backbone=backbone, max_detections=max_detections, mode="train", freeze_bn=freeze_backbone)

        if freeze_backbone:
            for i in range(190):
                print(model.layers[i].name)
                model.layers[i].trainable = False
        else:
            model.load_weights(ckpt_path, by_name=True, skip_mismatch=True)

            for layer in model.layers:
                layer.trainable = True

        model.compile(optimizer = optimizer, loss = {'centernet_loss': lambda y_true, y_pred: y_pred})
        # prediction_model.summary()
    
    # model.fit(train_dataset,
    #           steps_per_epoch = train_steps,
    #           validation_data = test_dataset,
    #           validation_steps = test_steps,
    #           callbacks = callbacks,
    #           epochs = epochs)

    model.fit(train_generator,
              steps_per_epoch = int(tf.math.ceil(train_generator.size() / batch_size).numpy()),
              validation_data = test_generator,
              validation_steps = int(tf.math.ceil(test_generator.size() / batch_size).numpy()),
              callbacks = callbacks,
              epochs = epochs)