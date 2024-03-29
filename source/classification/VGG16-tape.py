import pathlib, random, cv2
import tensorflow as tf
import numpy as np

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
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    # image = tf.cast(image, tf.float32) / 255.0
    image = (tf.cast(image, tf.float32) / 127.5) - 1
    # image = tf.image.per_image_standardization(image)

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

class VGG16(tf.keras.Model):
    def __init__(self):
        super(VGG16, self).__init__()        
        self.block1_conv1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', kernel_initializer='he_normal', bias_initializer='he_normal',)
        self.block1_conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', kernel_initializer='he_normal', bias_initializer='he_normal',)
        self.block1_pool = tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2), name='block1_pool')

        self.block2_conv1 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', kernel_initializer='he_normal', bias_initializer='he_normal',)
        self.block2_conv2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', kernel_initializer='he_normal', bias_initializer='he_normal',)
        self.block2_pool = tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2), name='block2_pool')

        self.block3_conv1 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', kernel_initializer='he_normal', bias_initializer='he_normal',)
        self.block3_conv2 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', kernel_initializer='he_normal', bias_initializer='he_normal',)
        self.block3_conv3 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', kernel_initializer='he_normal', bias_initializer='he_normal',)
        self.block3_pool = tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2), name='block3_pool')

        self.block4_conv1 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', kernel_initializer='he_normal', bias_initializer='he_normal',)
        self.block4_conv2 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', kernel_initializer='he_normal', bias_initializer='he_normal',)
        self.block4_conv3 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', kernel_initializer='he_normal', bias_initializer='he_normal',)
        self.block4_pool = tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2), name='block4_pool')

        self.block5_conv1 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', kernel_initializer='he_normal', bias_initializer='he_normal',)
        self.block5_conv2 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', kernel_initializer='he_normal', bias_initializer='he_normal',)
        self.block5_conv3 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', kernel_initializer='he_normal', bias_initializer='he_normal',)
        self.block5_pool = tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2), name='block5_pool')

        self.flatten = tf.keras.layers.Flatten(name='flatten')
        self.fc1 = tf.keras.layers.Dense(4096, activation='relu', name='fc1', kernel_initializer='he_normal', bias_initializer='he_normal',)
        self.dp1 = tf.keras.layers.Dropout(rate=0.5)
        self.fc2 = tf.keras.layers.Dense(4096, activation='relu', name='fc2', kernel_initializer='he_normal', bias_initializer='he_normal',)
        self.dp2 = tf.keras.layers.Dropout(rate=0.5)
        self.prediction = tf.keras.layers.Dense(n_classes, activation='softmax', kernel_initializer='he_normal', bias_initializer='he_normal', name='predictions')

    def call(self, inputs, training):
        net = self.block1_conv1(inputs)
        net = self.block1_conv2(net)
        net = self.block1_pool(net)
        net = self.block2_conv1(net)
        net = self.block2_conv2(net)
        net = self.block2_pool(net)
        net = self.block3_conv1(net)
        net = self.block3_conv2(net)
        net = self.block3_conv3(net)
        net = self.block3_pool(net)
        net = self.block4_conv1(net)
        net = self.block4_conv2(net)
        net = self.block4_conv3(net)
        net = self.block4_pool(net)
        net = self.block5_conv1(net)
        net = self.block5_conv2(net)
        net = self.block5_conv3(net)
        net = self.block5_pool(net)
        net = self.flatten(net)
        net = self.fc1(net)        
        net = self.dp1(net)            
        net = self.fc2(net)
        net = self.dp2(net)
        net = self.prediction(net)

        return net

@tf.function
def train(model, images, labels):
    with tf.GradientTape() as tape:
        y_pred = model(images, training=True)
        loss = tf.reduce_mean(cost_fn(labels, y_pred))
    
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.trainable_variables))

    train_acc.update_state(labels, y_pred)
    train_loss.update_state(labels, y_pred)


@tf.function
def validation(model, images, labels):
    y_pred = model(images, training=False)
    loss = tf.reduce_mean(cost_fn(labels, y_pred))
    
    val_acc.update_state(labels, y_pred)
    val_loss.update_state(labels, y_pred)

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
    EPOCHS = 100

   # learning rate scheduler
    LR_START = 0.001
    LR_MAX = 0.005 * strategy.num_replicas_in_sync
    LR_MIN = 0.001
    LR_RAMPUP_EPOCHS = 5
    LR_SUSTAIN_EPOCHS = 0
    LR_EXP_DECAY = .8
    
    # early stopping
    PATIENCE = 3
    EARLY_STOPPING = False
    minimum_loss = float(2147000000)

    train_dataset, total_train, n_classes = get_dataset('/home/v100/tf_workspace/Auged_datasets/natural_images/2021.04.08_08-51-48/train', True)
    test_dataset, total_valid, _ = get_dataset('/home/v100/tf_workspace/Auged_datasets/natural_images/2021.04.08_08-51-48/valid', False)
    n_classes = len(n_classes)

    cost_fn = tf.keras.losses.CategoricalCrossentropy()
    # lr_decay = tf.keras.optimizers.schedules.ExponentialDecay(0.00001, (total_train / BATCH_SIZE), 0.5, staircase=True)
    # optimizer = tf.keras.optimizers.Adam(learning_rate=lrfn)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001) # lr_decay
    inputs = tf.keras.Input(shape=(INPUT_SHAPE))
    model = VGG16()
    model(inputs=inputs)
    model.summary()
    # tf.keras.utils.plot_model(model, show_shapes=True)

    train_acc = tf.metrics.CategoricalAccuracy()
    train_loss = tf.metrics.CategoricalCrossentropy()
    val_acc = tf.metrics.CategoricalAccuracy()
    val_loss = tf.metrics.CategoricalCrossentropy()

    print()
    print('Learning started. It takes sometime.')
    stateful_matrices = ['train_acc', 'train_loss', 'valid_acc', 'valid_loss']
    for epoch in range(EPOCHS):
        print("Current Learning Rate : ", optimizer._decayed_lr('float32').numpy())
        tf.print("Epoch {}/{}".format(epoch + 1, EPOCHS))
        prog_bar = tf.keras.utils.Progbar(target=(total_train // BATCH_SIZE), stateful_metrics=stateful_matrices)

        train_acc.reset_states()
        train_loss.reset_states()
        val_acc.reset_states()
        val_loss.reset_states()
        
        for idx, (images, labels) in enumerate(train_dataset):
            train(model, images, labels)
            values=[('train_loss', train_loss.result().numpy()), ('train_acc', train_acc.result().numpy())]
            prog_bar.update(idx, values=values)

            if idx+1 >= (total_train // BATCH_SIZE):
                break

        for images, labels in test_dataset:
            validation(model, images, labels)

            if idx+1 >= (total_valid // BATCH_SIZE):
                break
        
        values = [('train_loss', train_loss.result().numpy()), ('train_acc', train_acc.result().numpy()), ('valid_loss', val_loss.result().numpy()), ('valid_acc', val_acc.result().numpy())]
        prog_bar.update((total_train // BATCH_SIZE), values=values, finalize=True)

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
    model.save('/home/v100/tf_workspace/model/vgg16')