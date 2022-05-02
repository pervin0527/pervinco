import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import tensorflow.keras.backend as K
import tensorflow_advanced_segmentation_models as tasm

from PIL import Image

def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()

def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    # for yet unknown reasons 0s and 1s need to be inverted... something is not working properly in the training pipeline
    # that is why iou_score and val_iou_score are so low while fitting the model...
    pred_mask = pred_mask == 0
    pred_mask = tf.cast(pred_mask, tf.float32)
    return pred_mask[0]

def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0

    return input_image, input_mask

@tf.function
def load_image_train(datapoint):
    input_image = tf.image.resize(datapoint['image'], (HEIGHT, WIDTH))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (HEIGHT, WIDTH))

    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask

def load_image_test(datapoint):
  input_image = tf.image.resize(datapoint['image'], (HEIGHT, WIDTH))
  input_mask = tf.image.resize(datapoint['segmentation_mask'], (HEIGHT, WIDTH))

  input_image, input_mask = normalize(input_image, input_mask)

  return input_image, input_mask


if __name__ == "__main__":
    dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)

    TRAIN_LENGTH = info.splits['train'].num_examples
    EPOCHS = 100
    BATCH_SIZE = 1
    BUFFER_SIZE = 1000
    HEIGHT, WIDTH = 320, 320
    STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE
    N_CLASSES = 2
    HEIGHT = 320
    WIDTH = 320
    BACKBONE_NAME = "efficientnetb5"
    WEIGHTS = "imagenet"

    train = dataset['train'].map(load_image_train, num_parallel_calls=tf.data.AUTOTUNE)
    test = dataset['test'].map(load_image_test)

    train_dataset = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    test_dataset = test.batch(BATCH_SIZE)

    for image, mask in train.take(5):
        sample_image, sample_mask = image, mask
        print(sample_mask.shape)
        break
        # display([sample_image, sample_mask])

    # base_model, layers, layer_names = tasm.create_base_model(name=BACKBONE_NAME, weights=WEIGHTS, height=HEIGHT, width=WIDTH, include_top=False, pooling=None)

    # BACKBONE_TRAINABLE = False
    # model = tasm.ACFNet(n_classes=N_CLASSES, base_model=base_model, output_layers=layers, backbone_trainable=BACKBONE_TRAINABLE)

    # opt = tf.keras.optimizers.SGD(learning_rate=0.2, momentum=0.9)

    # metrics = [tasm.metrics.IOUScore(threshold=0.5)]
    # categorical_focal_dice_loss = tasm.losses.CategoricalFocalLoss(alpha=0.25, gamma=2.0) + tasm.losses.DiceLoss()

    # model.compile(
    #     optimizer=opt,
    #     loss=categorical_focal_dice_loss,
    #     metrics=metrics,
    # )
    # model.run_eagerly = True

    # callbacks = [
    #             tf.keras.callbacks.ReduceLROnPlateau(monitor="iou_score", factor=0.2, patience=7, verbose=1, mode="max"),
    #             tf.keras.callbacks.EarlyStopping(monitor="iou_score", patience=19, mode="max", verbose=1, restore_best_weights=True)
    # ]

    # for layer in model.layers:
    #     if "model" in layer.name:
    #         layer.trainable = False

    #     print(layer.name + ": " + str(layer.trainable))

    # VAL_SUBSPLITS = 5
    # VALIDATION_STEPS = info.splits['test'].num_examples // BATCH_SIZE // VAL_SUBSPLITS

    # history = model.fit(
    #     train_dataset,
    #     epochs=EPOCHS,
    #     steps_per_epoch=STEPS_PER_EPOCH,
    #     # validation_steps=VALIDATION_STEPS,
    #     # validation_data=test_dataset,
    #     callbacks=callbacks
    #     )