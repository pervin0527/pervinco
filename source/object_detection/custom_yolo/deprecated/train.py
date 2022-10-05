import os
import yaml
import tensorflow as tf
import dataset as dataset
import tensorflow_addons as tfa

from glob import glob
from utils import freeze_all
from IPython.display import clear_output
from models import YoloV3, YoloV3Tiny, YoloLoss, yolo_anchors, yolo_anchor_masks,yolo_tiny_anchors, yolo_tiny_anchor_masks

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


def setup_model():
    if config["tiny"]:
        model = YoloV3Tiny(config["size"], training=True, classes=config["num_classes"], max_detections=config["max_detections"])
        anchors = yolo_tiny_anchors
        anchor_masks = yolo_tiny_anchor_masks
    else:
        model = YoloV3(config["size"], training=True, classes=config["num_classes"], max_detections=config["max_detections"])
        anchors = yolo_anchors
        anchor_masks = yolo_anchor_masks

    # Configure the model for transfer learning
    if config["transfer"] == 'none':
        pass  # Nothing to do
    elif config["transfer"] in ['darknet', 'no_output']:
        # Darknet transfer is a special case that works
        # with incompatible number of classes
        # reset top layers
        if config["tiny"]:
            model_pretrained = YoloV3Tiny(config["size"], training=True, classes=config["weights_num_classes"] or config["num_classes"])
        else:
            model_pretrained = YoloV3(config["size"], training=True, classes=config["weights_num_classes"] or config["num_classes"])
        model_pretrained.load_weights(config["weights"])

        if config["transfer"] == 'darknet':
            model.get_layer('yolo_darknet').set_weights(model_pretrained.get_layer('yolo_darknet').get_weights())
            freeze_all(model.get_layer('yolo_darknet'))
        elif config["transfer"] == 'no_output':
            for l in model.layers:
                if not l.name.startswith('yolo_output'):
                    l.set_weights(model_pretrained.get_layer(l.name).get_weights())
                    freeze_all(l)
    else:
        # All other transfer require matching classes
        model.load_weights(config["weights"])
        if config["transfer"] == 'fine_tune':
            # freeze darknet and fine tune other layers
            darknet = model.get_layer('yolo_darknet')
            freeze_all(darknet)
        elif config["transfer"] == 'frozen':
            # freeze everything
            freeze_all(model)

    # optimizer = tf.keras.optimizers.Adam(learning_rate=config["learning_rate"])
    optimizer = tfa.optimizers.AdamW(learning_rate=config["learning_rate"], weight_decay=config["weight_decay"])
    loss = [YoloLoss(anchors[mask], classes=config["num_classes"])
            for mask in anchor_masks]

    model.compile(optimizer=optimizer, loss=loss)

    return model, optimizer, loss, anchors, anchor_masks


def plot_predictions(model):
    images = sorted(glob("./test_imgs/*.jpg"))
    for file in images:
        image = tf.image.decode_image(open(file, "rb").read(), channels=3)
        image = tf.image.resize(image, (config["size"], config["size"]))
        input_tensor = tf.expand_dims(image, axis=0)
        input_tensor = input_tensor / 255.

        predictions = model(input_tensor)
        print(predictions)


class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)

        if not config["tiny"]:
            pred_model = YoloV3(classes=config["num_classes"])
            pred_model.load_weights("/data/Models/custom_yolo/train.tf", by_name=True, skip_mismatch=True)
        
        plot_predictions(model=pred_model)


if __name__ == '__main__':
    with open("./config.yaml", "r") as f:
        config = yaml.safe_load(f)

    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    BATCH_SIZE = config["batch_size"] * strategy.num_replicas_in_sync

    with strategy.scope():
        model, optimizer, loss, anchors, anchor_masks = setup_model()

    train_dataset = dataset.load_tfrecord_dataset(config["train_dataset"], config["classes"], config["size"], config["max_detections"])
    train_dataset = train_dataset.shuffle(buffer_size=512)
    train_dataset = train_dataset.batch(BATCH_SIZE)
    train_dataset = train_dataset.map(lambda x, y: (dataset.transform_images(x, config["size"]), dataset.transform_targets(y, anchors, anchor_masks, config["size"])))
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    val_dataset = dataset.load_tfrecord_dataset(config["valid_dataset"], config["classes"], config["size"])
    val_dataset = val_dataset.batch(BATCH_SIZE)
    val_dataset = val_dataset.map(lambda x, y: (dataset.transform_images(x, config["size"]), dataset.transform_targets(y, anchors, anchor_masks, config["size"])))

    clr = tfa.optimizers.CyclicalLearningRate(initial_learning_rate=config["learning_rate"],
                                              maximal_learning_rate=config["max_lr"],
                                              scale_fn=lambda x : 1.0,
                                              step_size=config["epochs"] / 2)

    callbacks = [
        tf.keras.callbacks.LearningRateScheduler(clr),
        tf.keras.callbacks.ModelCheckpoint('/data/Models/custom_yolo/yolov3_train_{epoch}.tf', verbose=1, save_weights_only=True, save_best_only=True),
        # DisplayCallback(),
    ]

    history = model.fit(train_dataset,
                        epochs=config["epochs"],
                        callbacks=callbacks,
                        validation_data=val_dataset)