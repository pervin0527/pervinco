import pathlib, random, cv2
import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K

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

def preprocess_image(images, label):
    image = tf.io.read_file(images)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.per_image_standardization(image)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])

    return image, label

def get_dataset(ds_path, is_train):
    ds_path = pathlib.Path(ds_path)

    images = list(ds_path.glob('*/*.jpg'))
    images = [str(path) for path in images]
    total_images = len(images)

    if is_train:
        random.shuffle(images)

    labels = sorted(item.name for item in ds_path.glob('*/') if item.is_dir())
    classes = labels
    labels = dict((name, index) for index, name in enumerate(labels))
    labels = [labels[pathlib.Path(path).parent.name] for path in images]
    labels = tf.keras.utils.to_categorical(labels, num_classes=len(classes), dtype='float32')

    if is_train:
        dataset = tf.data.Dataset.from_tensor_slices((images, labels))
        dataset = dataset.map(preprocess_image, num_parallel_calls=AUTOTUNE)
        dataset = dataset.shuffle(512)
        dataset = dataset.batch(BATCH_SIZE)
        dataset = dataset.prefetch(AUTOTUNE)
    
    else:
        dataset = tf.data.Dataset.from_tensor_slices((images, labels))
        dataset = dataset.map(preprocess_image, num_parallel_calls=AUTOTUNE)
        dataset = dataset.batch(BATCH_SIZE)
        dataset = dataset.prefetch(AUTOTUNE)

    return dataset, total_images, classes

def residual_block(x, filters, kernel_size=3, stride=1, conv_shortcut=True, name=None):
    if conv_shortcut:
        shortcut = tf.keras.layers.Conv2D(4 * filters, 1, strides=stride, name=name+'_0_conv')(x)
        shortcut = tf.keras.layers.BatchNormalization(axis=3, epsilon=1.001e-5, name=name+'_0_bn')(shortcut)

    else:
        shortcut = x

    x = tf.keras.layers.Conv2D(filters, 1, strides=stride, name=name + '_1_conv')(x)
    x = tf.keras.layers.BatchNormalization(axis=3, epsilon=1.001e-5, name=name + '_1_bn')(x)
    x = tf.keras.layers.Activation('relu', name=name + '_1_relu')(x)

    x = tf.keras.layers.Conv2D(filters, kernel_size, padding='SAME', name=name + '_2_conv')(x)
    x = tf.keras.layers.BatchNormalization(axis=3, epsilon=1.001e-5, name=name + '_2_bn')(x)
    x = tf.keras.layers.Activation('relu', name=name + '_2_relu')(x)

    x = tf.keras.layers.Conv2D(4 * filters, 1, name=name + '_3_conv')(x)
    x = tf.keras.layers.BatchNormalization(axis=3, epsilon=1.001e-5, name=name + '_3_bn')(x)

    x = tf.keras.layers.Add(name=name + '_add')([shortcut, x])
    x = tf.keras.layers.Activation('relu', name=name + '_out')(x)

    return x

def residual_stack(x, filters, blocks, stride1=2, name=None):
    x = residual_block(x, filters, stride=stride1, name=name + '_block1')

    for i in range(2, blocks + 1):
        x = residual_block(x, filters, conv_shortcut=False, name=name + '_block' + str(i))

    return x

def ResNet50():
    inputs = tf.keras.layers.Input(shape=INPUT_SHAPE)
    x = tf.keras.layers.ZeroPadding2D(padding=((3, 3), (3, 3)), name='conv1_pad')(inputs)
    x = tf.keras.layers.Conv2D(64, 7, strides=2, use_bias=True, name='conv1_conv')(x)
    x = tf.keras.layers.BatchNormalization(axis=3, epsilon=1.001e-5, name='conv1_bn')(x)
    x = tf.keras.layers.Activation('relu', name='conv1_relu')(x)

    x = tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad')(x)
    x = tf.keras.layers.MaxPooling2D(3, strides=2, name='pool1_pool')(x)

    x = residual_stack(x, 64, 3, stride1=1, name='conv2')
    x = residual_stack(x, 128, 4, name='conv3')
    x = residual_stack(x, 256, 6, name='conv4')
    x = residual_stack(x, 512, 3, name='conv5')

    x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
    outputs = tf.keras.layers.Dense(n_classes, activation='softmax', name='predictions')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model

@tf.function
def train(model, images, labels):
    with tf.GradientTape() as tape:
        y_pred = model(images, training=True)
        loss = LOSS(labels, y_pred)
        grads = tape.gradient(loss, model.trainable_variables)
    
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.trainable_variables))
    train_acc.update_state(labels, y_pred)
    train_loss.update_state(loss)

@tf.function
def validation(model, images, labels):
    y_pred = model(images, training=False)
    loss = LOSS(labels, y_pred)
    val_acc.update_state(labels, y_pred)
    val_loss.update_state(loss)

def lrfn():
    if epoch < LR_RAMPUP_EPOCHS:
        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
        lr = LR_MAX
    else:
        lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY**(epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN
    return lr

if __name__ == "__main__":
    # hyper parameters
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    IMG_SIZE = 224
    INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
    BATCH_SIZE = 32
    EPOCHS = 1000

    # learning rate scheduler
    LR_START = 0.0001
    LR_MAX = 0.0005 * strategy.num_replicas_in_sync
    LR_MIN = 0.0001
    LR_RAMPUP_EPOCHS = 5
    LR_SUSTAIN_EPOCHS = 0
    LR_EXP_DECAY = .8
    
    # early stopping
    PATIENCE = 3
    EARLY_STOPPING = True
    minimum_loss = float(2147000000)

    train_dataset, total_train, n_classes = get_dataset('/data/backup/pervinco/Auged_datasets/natural_images/2021.03.26_09:26:52/train', True)
    test_dataset, total_valid, _ = get_dataset('/data/backup/pervinco/Auged_datasets/natural_images/2021.03.26_09:26:52/valid', False)
    n_classes = len(n_classes)

    LOSS = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.SGD(learning_rate=lrfn)
    inputs = tf.keras.Input(shape=(INPUT_SHAPE))
    model = ResNet50()
    model(inputs=inputs)
    model.summary()
    tf.keras.utils.plot_model(model, show_shapes=True)

    train_acc = tf.metrics.CategoricalAccuracy()
    train_loss = tf.metrics.Mean()
    val_acc = tf.metrics.CategoricalAccuracy()
    val_loss = tf.metrics.Mean()

    print()
    print('Learning started. It takes sometime.')
    for epoch in range(EPOCHS):
        print("Current Learning Rate : ", optimizer._decayed_lr('float32').numpy())
        tf.print("Epoch {}/{}".format(epoch + 1, EPOCHS))
        prog_bar = tf.keras.utils.Progbar(target=total_train)

        train_acc.reset_states()
        train_loss.reset_states()
        val_acc.reset_states()
        val_loss.reset_states()
        
        for idx, (images, labels) in enumerate(train_dataset):
            train(model, images, labels)
            values=[('train_loss', train_loss.result().numpy()), ('train_acc', train_acc.result().numpy())]
            prog_bar.update(BATCH_SIZE*idx, values=values)

        for images, labels in test_dataset:
            validation(model, images, labels)
        
        values = [('train_loss', train_loss.result().numpy()), ('train_acc', train_acc.result().numpy()), ('valid_loss', val_loss.result().numpy()), ('valid_acc', val_acc.result().numpy())]
        prog_bar.update(total_train, values=values, finalize=True)

        if EARLY_STOPPING:
            tmp_loss = (val_loss.result().numpy())
            if tmp_loss < minimum_loss:
                minimum_loss = tmp_loss
                PATIENCE = 3

            else:
                PATIENCE -= 1

                if PATIENCE == 0:
                    break

    print('Learning Finished')
    model.save('/data/backup/pervinco/model/test_model')