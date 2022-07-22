import cv2
import random
import pathlib
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from glob import glob
from datetime import datetime
from tensorflow.keras.preprocessing.image import img_to_array

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

def load_label(path):
    df = pd.read_csv(path, sep=',', index_col=False, header=None)
    labels = df[0].tolist()
    
    return labels

def build_model():
    effnet = tf.keras.applications.EfficientNetB1(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    model = effnet.output
    model = tf.keras.layers.GlobalAveragePooling2D()(model)
    model = tf.keras.layers.Dropout(0.5)(model)
    model = tf.keras.layers.Dense(len(classes), activation='softmax')(model)

    model = tf.keras.Model(inputs=effnet.input, outputs=model)
    model.summary()

    return model

def preprocess_image(images, label=None):
    image = tf.io.read_file(images)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.keras.applications.efficientnet.preprocess_input(image)

    if label is None:
        return image

    else:
        return image, label

def build_tf_dataset(path, train):
    ds = pathlib.Path(path)
    images = list(ds.glob('*/*'))
    images = [str(path) for path in images]
    total_images = len(images)

    if train:
        random.shuffle(images)

    labels = sorted(item.name for item in ds.glob('*/') if item.is_dir())
    classes = labels
    labels = dict((name, index) for index, name in enumerate(labels))
    labels = [labels[pathlib.Path(path).parent.name] for path in images]
    labels = tf.keras.utils.to_categorical(labels, num_classes=len(classes), dtype='float32')

    if train:
        dataset = (tf.data.Dataset
                   .from_tensor_slices((images, labels))
                   .map(preprocess_image, num_parallel_calls=AUTO)
                   .repeat()
                   .shuffle(512)
                   .batch(BATCH_SIZE)
                   .prefetch(AUTO)
        )
    
    else:
        dataset = (tf.data.Dataset
                   .from_tensor_slices((images, labels))
                   .map(preprocess_image, num_parallel_calls=AUTO)
                   .repeat()
                   .batch(BATCH_SIZE)
                   .prefetch(AUTO)
        )

    return dataset, total_images, classes

def build_lrfn(lr_start=0.00001, lr_max=0.00005, 
               lr_min=0.00001, lr_rampup_epochs=5, 
               lr_sustain_epochs=0, lr_exp_decay=.8):
    lr_max = lr_max * strategy.num_replicas_in_sync

    def lrfn(epoch):
        if epoch < lr_rampup_epochs:
            lr = (lr_max - lr_start) / lr_rampup_epochs * epoch + lr_start
        elif epoch < lr_rampup_epochs + lr_sustain_epochs:
            lr = lr_max
        else:
            lr = (lr_max - lr_min) *\
                 lr_exp_decay**(epoch - lr_rampup_epochs\
                                - lr_sustain_epochs) + lr_min
        return lr
    return lrfn

def VizGradCAM(model, image, interpolant=0.5, plot_results=True):
    original_img = np.asarray(image, dtype = np.float32)
    img = np.expand_dims(original_img, axis=0)
    prediction = model.predict(img)
    prediction_idx = np.argmax(prediction)

    last_conv_layer = next(x for x in model.layers[::-1] if isinstance(x, tf.keras.layers.Conv2D))
    target_layer = model.get_layer(last_conv_layer.name)

    with tf.GradientTape() as tape:
        gradient_model = tf.keras.Model([model.inputs], [target_layer.output, model.output])
        conv2d_out, prediction = gradient_model(img)
        loss = prediction[:, prediction_idx]

    gradients = tape.gradient(loss, conv2d_out)
    output = conv2d_out[0]
    weights = tf.reduce_mean(gradients[0], axis=(0, 1))
    activation_map = np.zeros(output.shape[0:2], dtype=np.float32)
    for idx, weight in enumerate(weights):
        activation_map += weight * output[:, :, idx]
    activation_map = cv2.resize(activation_map.numpy(), 
                                (original_img.shape[1], 
                                 original_img.shape[0]))
    activation_map = np.maximum(activation_map, 0)
    activation_map = (activation_map - activation_map.min()) / (activation_map.max() - activation_map.min())
    activation_map = np.uint8(255 * activation_map)


    heatmap = cv2.applyColorMap(activation_map, cv2.COLORMAP_JET)

    #superimpose heatmap onto image
    original_img = np.uint8((original_img - original_img.min()) / (original_img.max() - original_img.min()) * 255)
    cvt_heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    cvt_heatmap = img_to_array(cvt_heatmap)

    plt.imshow(np.uint8(original_img * interpolant + cvt_heatmap * (1 - interpolant)))
    plt.show()
    # cv2.imshow('result', cvt_heatmap)
    # cv2.waitKey(0)

    # #enlarge plot
    # plt.rcParams["figure.dpi"] = 100

    # if plot_results == True:
    #     plt.imshow(np.uint8(original_img * interpolant + cvt_heatmap * (1 - interpolant)))
    # else:
    #     return cvt_heatmap


if __name__ == "__main__":
    label_path = "/data/Datasets/SPC/Labels/labels.txt"
    dataset_path ="/data/Datasets/SPC/Cls"
    testset_path = "/data/Datasets/SPC/Testset/Normal/images"
    save_path = f"/data/Models/classification/SPC"
# 
    EPOCHS = 300
    BATCH_SIZE = 32
    IMG_SIZE = 320
    LR = 0.000001
    checkpoint_path = "/data/Models/classification/SPC/"
    AUTO = tf.data.experimental.AUTOTUNE

    classes = load_label(label_path)
    print(classes)
    train_data, len_train_data, train_classes = build_tf_dataset(f"{dataset_path}/train2", True)
    valid_data, len_valid_data, valid_classes = build_tf_dataset(f"{dataset_path}/valid2", False)
    TRAIN_STEPS_PER_EPOCH = int(tf.math.ceil(len_train_data/ BATCH_SIZE).numpy())
    VALID_STEP_PER_EPOCH = int(tf.math.ceil(len_valid_data / BATCH_SIZE).numpy())

    model = build_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR), loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    now = datetime.now().strftime("%Y.%m.%d_%H:%M")
    checkpoint = tf.keras.callbacks.ModelCheckpoint(f"{save_path}/{now}/", save_weights_only=True)
    earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, mode='auto', verbose=1)
    # reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, min_delta=0.001, mode='auto', verbose=1)
    lrfn = build_lrfn()
    cosine_lr = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=1)


    history = model.fit(train_data,
                        epochs=EPOCHS,
                        steps_per_epoch=TRAIN_STEPS_PER_EPOCH,
                        verbose=1,
                        validation_data=valid_data,
                        validation_steps=VALID_STEP_PER_EPOCH,
                        callbacks=[checkpoint, earlystop, cosine_lr])

    # model = tf.keras.models.load_model()
    tf.saved_model.save(model, f'{save_path}/{now}')

    # model = tf.keras.models.load_model("/data/Models/classification/SPC/2022.04.11_09:39")    
    model = tf.keras.models.load_model(f"{save_path}/{now}")
    test_img = cv2.imread("/data/Datasets/SPC/Testset/Normal/images/0001.jpg")
    test_img = cv2.resize(test_img, (IMG_SIZE, IMG_SIZE))
    VizGradCAM(model, img_to_array(test_img), plot_results=True)