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

def preprocess_image(images, label=None):
    image = tf.io.read_file(images)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    # image = tf.keras.applications.efficientnet.preprocess_input(image)

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
        self.block1_conv1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', kernel_initializer='he_uniform')
        self.block1_conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', kernel_initializer='he_uniform')
        self.block1_pool = tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2), name='block1_pool')

        self.block2_conv1 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', kernel_initializer='he_uniform')
        self.block2_conv2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', kernel_initializer='he_uniform')
        self.block2_pool = tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2), name='block2_pool')

        self.block3_conv1 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', kernel_initializer='he_uniform')
        self.block3_conv2 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', kernel_initializer='he_uniform')
        self.block3_conv3 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', kernel_initializer='he_uniform')
        self.block3_pool = tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2), name='block3_pool')

        self.block4_conv1 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', kernel_initializer='he_uniform')
        self.block4_conv2 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', kernel_initializer='he_uniform')
        self.block4_conv3 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', kernel_initializer='he_uniform')
        self.block4_pool = tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2), name='block4_pool')

        self.block5_conv1 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', kernel_initializer='he_uniform')
        self.block5_conv2 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', kernel_initializer='he_uniform')
        self.block5_conv3 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', kernel_initializer='he_uniform')
        self.block5_pool = tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2), name='block5_pool')

        self.flatten = tf.keras.layers.Flatten(name='flatten')
        self.fc1 = tf.keras.layers.Dense(4096, activation='relu', name='fc1', kernel_initializer='he_uniform')
        self.dp1 = tf.keras.layers.Dropout(rate=0.5)
        self.fc2 = tf.keras.layers.Dense(4096, activation='relu', name='fc2', kernel_initializer='he_uniform')
        self.dp2 = tf.keras.layers.Dropout(rate=0.5)
        self.prediction = tf.keras.layers.Dense(n_classes, activation='softmax', name='predictions')

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
def loss_fn(model, images, labels):
    logits = model(images, training=True)
    loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_pred=logits, y_true=labels))

    return loss

@tf.function
def grad(model, images, labels):
    with tf.GradientTape() as tape:
        loss = loss_fn(model, images, labels)

    return tape.gradient(loss, model.variables)

@tf.function
def evaluate(model, images, labels):
    logits = model(images, training=False)
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy

@tf.function
def train(model, images, labels):
    grads = grad(model, images, labels)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

if __name__ == "__main__":
    IMG_SIZE = 224
    BATCH_SIZE = 32
    LR_INIT = 0.00001
    EPOCHS = 200
    minimum_loss = 2147000000    
    PATIENCE = 3
    INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    train_dataset, _, n_classes = get_dataset('/data/tf_workspace/Auged_datasets/natural_images/2021.03.26_09:26:52/train', True)
    test_dataset, _, _ = get_dataset('/data/tf_workspace/Auged_datasets/natural_images/2021.03.26_09:26:52/valid', False)
    n_classes = len(n_classes)

    optimizer = tf.keras.optimizers.SGD(learning_rate=LR_INIT)
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    model = VGG16()
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    # model.summary()
    
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)]
    model.fit(train_dataset,
              epochs=EPOCHS,
              verbose=1,
              validation_data=test_dataset,
              callbacks=callbacks)