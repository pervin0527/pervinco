import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import pandas as pd
import tensorflow as tf
import albumentations as A
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
import tensorflow_advanced_segmentation_models as tasm

# GPU setup
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


def get_dataset_counts(d):
    pixel_count = np.array([i for i in d.values()])

    sum_pixel_count = 0
    for i in pixel_count:
        sum_pixel_count += i

    return pixel_count, sum_pixel_count


def get_dataset_statistics(pixel_count, sum_pixel_count):
    
    pixel_frequency = np.round(pixel_count / sum_pixel_count, 4)

    mean_pixel_frequency = np.round(np.mean(pixel_frequency), 4)

    return pixel_frequency, mean_pixel_frequency


def get_balancing_class_weights(classes, d):
    pixel_count, sum_pixel_count = get_dataset_counts(d)

    background_pixel_count = 0
    mod_pixel_count = []

    for c in TOTAL_CLASSES:
        if c not in classes:
            background_pixel_count += d[c]
        else:
            mod_pixel_count.append(d[c])

    if not ALL_CLASSES:
        mod_pixel_count.append(background_pixel_count)
    else:
        mod_pixel_count[:-1]
    
    pixel_frequency, mean_pixel_frequency = get_dataset_statistics(mod_pixel_count, sum_pixel_count)

    class_weights = np.round(mean_pixel_frequency / pixel_frequency, 2)
    return class_weights    


def round_clip_0_1(x, **kwargs):
    return x.round().clip(0, 1)


def get_training_augmentation(height, width):
    train_transform = [
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(scale_limit=0.6, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
        A.PadIfNeeded(min_height=height, min_width=width, always_apply=True, border_mode=0),
        A.RandomCrop(height=height, width=width, always_apply=True),
        # A.IAAAdditiveGaussianNoise(p=0.2),
        # A.IAAPerspective(p=0.5),

        A.OneOf([A.CLAHE(p=1),
                 A.RandomBrightnessContrast(p=1),
                 A.HueSaturationValue(p=1),
                 A.RandomGamma(p=1),
            ], p=0.9),

        A.OneOf([
                # A.IAASharpen(p=1),
                A.Blur(blur_limit=3, p=1),
                A.MotionBlur(blur_limit=3, p=1),
        ],p=0.9),
        A.Lambda(mask=round_clip_0_1)
    ]
    return A.Compose(train_transform)


def get_validation_augmentation(height, width):
    test_transform = [
        A.PadIfNeeded(height, width),
        A.Resize(height, width, always_apply=True)
    ]
    return A.Compose(test_transform)

def create_image_label_path_generator(images_dir, masks_dir, shuffle=False, seed=None):
    ids = sorted(os.listdir(images_dir))
    mask_ids = sorted(os.listdir(masks_dir))

    if shuffle == True:

        if seed is not None:
            tf.random.set_seed(seed)

        indices = tf.range(start=0, limit=tf.shape(ids)[0], dtype=tf.int32)
        shuffled_indices = tf.random.shuffle(indices)

        ids = tf.gather(ids, shuffled_indices).numpy().astype(str)
        mask_ids = tf.gather(mask_ids, shuffled_indices).numpy().astype(str)

    images_fps = [os.path.join(images_dir, image_id) for image_id in ids]
    masks_fps = [os.path.join(masks_dir, image_id) for image_id in mask_ids]

    while True:
        for i in range(len(images_fps)):
            yield [images_fps[i], masks_fps[i]]


def process_image_label(images_paths, masks_paths, classes, augmentation=None, preprocessing=None, all_classes=False):
    class_values = [TOTAL_CLASSES.index(cls.lower()) for cls in classes]
    # print(class_values)
    
    image = cv2.imread(images_paths)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(masks_paths, 0)

    # extract certain classes from mask (e.g. cars)
    masks = [(mask == v) for v in class_values]
    mask = np.stack(masks, axis=-1).astype('float')
    
    # add background if mask is not binary
    if mask.shape[-1] != 1 and not all_classes:
        background = 1 - mask.sum(axis=-1, keepdims=True)
        mask = np.concatenate((mask, background), axis=-1)
    
    # apply augmentations
    if augmentation:
        sample = augmentation(image=image, mask=mask)
        image, mask = sample['image'], sample['mask']
    
    # apply preprocessing
    if preprocessing:
        sample = preprocessing(image=image, mask=mask)
        image, mask = sample['image'], sample['mask']

    # mask = np.squeeze(np.argmax(mask, axis=-1))
    # mask = np.argmax(mask, axis=-1)
    # mask = mask[..., np.newaxis]
        
    return image, mask


def DataGenerator(train_dir, label_dir, batch_size, height, width, classes, augmentation, all_classes=False, wwo_aug=False, shuffle=False, seed=None):
    image_label_path_generator = create_image_label_path_generator(train_dir, label_dir, shuffle=shuffle, seed=seed)

    if wwo_aug:
        while True:
            images = np.zeros(shape=[batch_size, height, width, 3])
            if all_classes:
                labels = np.zeros(shape=[batch_size, height, width, len(classes)], dtype=np.float32)
            else:
                labels = np.zeros(shape=[batch_size, height, width, len(classes) + 1], dtype=np.float32)
            for i in range(0, batch_size, 2):
                image_path, label_path = next(image_label_path_generator)
                image_aug, label_aug = process_image_label(image_path, label_path, classes=classes, augmentation=augmentation, all_classes=all_classes)
                image_wo_aug, label_wo_aug = process_image_label(image_path, label_path, classes=classes, augmentation=get_validation_augmentation(height=HEIGHT, width=WIDTH), all_classes=all_classes)
                images[i], labels[i] = image_aug, label_aug
                images[i + 1], labels[i + 1] = image_wo_aug, label_wo_aug

            yield tf.convert_to_tensor(images), tf.convert_to_tensor(labels, tf.float32)
    else:
        while True:
            images = np.zeros(shape=[batch_size, height, width, 3])
            if all_classes:
                labels = np.zeros(shape=[batch_size, height, width, len(classes)], dtype=np.float32)
            else:
                labels = np.zeros(shape=[batch_size, height, width, len(classes) + 1], dtype=np.float32)
            for i in range(batch_size):
                image_path, label_path = next(image_label_path_generator)
                image, label = process_image_label(image_path, label_path, classes=classes, augmentation=augmentation, all_classes=all_classes)
                images[i], labels[i] = image, label

            yield tf.convert_to_tensor(images), tf.convert_to_tensor(labels, tf.float32)


def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()


def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]


def show_predictions():
    gt_mask = onehot_decode(tf.cast(sample_mask, dtype=tf.uint8))
    pred = model.predict(sample_image[tf.newaxis, ...])
    pred_mask = create_mask(pred)
    print(sample_image.shape, sample_mask.shape, pred_mask.shape)

    display([sample_image, gt_mask, pred_mask])


def onehot_decode(input):
    stack = []
    for i, cls in enumerate(MODEL_CLASSES):
        idx = TOTAL_CLASSES.index(cls)
        img = tf.repeat(input[:, :, i, tf.newaxis], 3, axis=-1)
        img = img * tf.constant(VOC_COLORMAP[idx][::-1], dtype=tf.uint8)
        stack.append(img)
    stack = tf.stack(stack, axis=-1)
    return tf.reduce_sum(stack, axis=-1)


# def onehot_encode(input):
#     stack = []
#     for i in range(len(MODEL_CLASSES)):
#         stack.append(tf.math.reduce_prod(tf.cast(input == VOC_COLORMAP[i][::-1], tf.float32), axis=-1))
#     stack = tf.stack(stack, axis=-1)

class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        from IPython.display import clear_output

        clear_output(wait=True)
        show_predictions()
        print ('\n에포크 이후 예측 예시 {}\n'.format(epoch+1))


if __name__ == "__main__":
    DATA_DIR = "/data/Datasets/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/Segmentation"
    x_train_dir = f"{DATA_DIR}/train/images"
    y_train_dir = f"{DATA_DIR}/train/masks"
    x_valid_dir = f"{DATA_DIR}/valid/images"
    y_valid_dir = f"{DATA_DIR}/valid/masks"


    TOTAL_CLASSES = pd.read_csv(f"{DATA_DIR}/labels/class_labels.txt", sep='\n', header=None, index_col=False)
    TOTAL_CLASSES = TOTAL_CLASSES[0].to_list()
    N_TOTAL_CLASSES = len(TOTAL_CLASSES)
    
    CLASSES_PIXEL_COUNT_DICT = {'background': 361560627, 'aeroplane': 3704393, 'bicycle': 1571148, 'bird': 4384132, 'boat': 2862913,
                                'bottle': 3438963, 'bus': 8696374, 'car': 7088203, 'cat': 12473466,'chair': 4975284,
                                'cow': 5027769, 'diningtable': 6246382, 'dog': 9379340, 'horse': 4925676, 'motorbike': 5476081,
                                'person': 24995476, 'potted plant': 2904902, 'sheep': 4187268, 'sofa': 7091464, 'train': 7903243, 'tv/monitor': 4120989}

    MODEL_CLASSES = TOTAL_CLASSES
    ALL_CLASSES = False
    if MODEL_CLASSES == TOTAL_CLASSES:
        # MODEL_CLASSES = MODEL_CLASSES[:-1]
        # MODEL_CLASSES = MODEL_CLASSES[1:]
        ALL_CLASSES = True

    EPOCHS = 500
    BATCH_SIZE = 16
    N_CLASSES = len(MODEL_CLASSES)
    HEIGHT,WIDTH = 320, 320
    WEIGHTS = "imagenet"
    BACKBONE_NAME = "efficientnetb0"
    WWO_AUG = False

    VOC_COLORMAP = [[0, 0, 0],
                    [128, 0, 0],
                    [0, 128, 0],
                    [128, 128, 0],
                    [0, 0, 128],
                    [128, 0, 128],
                    [0, 128, 128],
                    [128, 128, 128],
                    [64, 0, 0],
                    [192, 0, 0],
                    [64, 128, 0],
                    [192, 128, 0],
                    [64, 0, 128],
                    [192, 0, 128],
                    [64, 128, 128],
                    [192, 128, 128],
                    [0, 64, 0],
                    [128, 64, 0],
                    [0, 192, 0],
                    [128, 192, 0],
                    [0, 64, 128]]

    class_weights = get_balancing_class_weights(MODEL_CLASSES, CLASSES_PIXEL_COUNT_DICT)
    print(class_weights)

    ## MODEL
    base_model, layers, layer_names = tasm.create_base_model(name=BACKBONE_NAME, weights=WEIGHTS, height=HEIGHT, width=WIDTH, include_top=False, pooling=None)
    BACKBONE_TRAINABLE = False
    model = tasm.DeepLabV3plus(n_classes=N_CLASSES, base_model=base_model, output_layers=layers, backbone_trainable=BACKBONE_TRAINABLE)

    opt = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)
    metrics = [tasm.metrics.IOUScore(threshold=0.5)]
    categorical_focal_dice_loss = tasm.losses.CategoricalFocalLoss(alpha=0.25, gamma=2.0) + tasm.losses.DiceLoss(class_weights=class_weights)

    model.compile(optimizer=opt, loss=categorical_focal_dice_loss, metrics=metrics)
    model.run_eagerly = True

    callbacks = [tf.keras.callbacks.ReduceLROnPlateau(monitor="iou_score", factor=0.2, patience=6, verbose=1, mode="max"),
                 tf.keras.callbacks.ModelCheckpoint("/data/Models/segmentation/DeepLabV3plus.ckpt", verbose=1, save_weights_only=True, save_best_only=True),
                #  DisplayCallback(),
                 tf.keras.callbacks.EarlyStopping(monitor="iou_score", patience=16, mode="max", verbose=1, restore_best_weights=True)]

    TrainSet = DataGenerator(x_train_dir,
                             y_train_dir,
                             BATCH_SIZE,
                             HEIGHT,
                             WIDTH,
                             classes=MODEL_CLASSES,
                             augmentation=get_training_augmentation(height=HEIGHT, width=WIDTH),
                             all_classes=ALL_CLASSES,
                             shuffle=True,
                             seed=None)

    ValidationSet = DataGenerator(x_valid_dir,
                                  y_valid_dir,
                                  1,
                                  HEIGHT,
                                  WIDTH,
                                  classes=MODEL_CLASSES,
                                  augmentation=get_validation_augmentation(height=HEIGHT, width=WIDTH),
                                  all_classes=ALL_CLASSES,
                                  shuffle=False,
                                  seed=None)

    TestSet = DataGenerator(x_valid_dir,
                            y_valid_dir,
                            1,
                            HEIGHT,
                            WIDTH,
                            classes=MODEL_CLASSES,
                            augmentation=get_validation_augmentation(height=HEIGHT, width=WIDTH),
                            all_classes=ALL_CLASSES,
                            shuffle=False)


    print(len(os.listdir(x_train_dir)))
    print(len(os.listdir(x_valid_dir)))
    print(len(os.listdir(x_valid_dir)))

    for i in TrainSet:
        sample_image, sample_mask = i[0][0], i[1][0]
        show_predictions()
        break

    # for layer in model.layers:
    #     if "model" in layer.name:
    #         layer.trainable = False
    #     print(layer.name + ": " + str(layer.trainable))

    for layer in model.layers:
        layer.trainable = True
        # print(layer.name + ": " + str(layer.trainable))

    if WWO_AUG:
        steps_per_epoch = np.floor(len(os.listdir(x_train_dir)) / BATCH_SIZE) * 2
        val_steps_per_epoch = np.floor(len(os.listdir(x_valid_dir)) / BATCH_SIZE) * 2
    else:
        steps_per_epoch = np.floor(len(os.listdir(x_train_dir)) / BATCH_SIZE)
        val_steps_per_epoch = np.floor(len(os.listdir(x_valid_dir)) / BATCH_SIZE)

    history = model.fit(TrainSet,
                        steps_per_epoch=steps_per_epoch,
                        epochs=EPOCHS,
                        callbacks=callbacks,
                        validation_data=ValidationSet,
                        validation_steps=val_steps_per_epoch)

    for idx, (image, mask) in enumerate(TestSet):
        sample_image, sample_mask = image[0], mask[0]
        show_predictions()

        if idx == 4:
            break