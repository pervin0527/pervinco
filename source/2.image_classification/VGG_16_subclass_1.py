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
        dataset = (tf.data.Dataset
                   .from_tensor_slices((images, labels))
                   .map(preprocess_image, num_parallel_calls=AUTOTUNE)
                #    .repeat()
                   .shuffle(512)
                   .batch(BATCH_SIZE)
                   .prefetch(AUTOTUNE)
        )
    
    else:
        dataset = (tf.data.Dataset
                   .from_tensor_slices((images, labels))
                   .map(preprocess_image, num_parallel_calls=AUTOTUNE)
                #    .repeat()
                   .batch(BATCH_SIZE)
                   .prefetch(AUTOTUNE)
        )

    return dataset, total_images, classes

class VGG16(tf.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        initializer = tf.keras.initializers.HeUniform()
        self.conv0_w = tf.Variable(initializer(shape=[3, 3, 3, 64]), trainable=True)
        self.conv0_b = tf.Variable(tf.constant(0.1, shape=[64]), trainable=True)
        self.conv1_w = tf.Variable(initializer(shape=[3, 3, 64, 64]), trainable=True)
        self.conv1_b = tf.Variable(tf.constant(0.1, shape=[64]), trainable=True)
        self.conv2_w = tf.Variable(initializer(shape=[3, 3, 64, 128]), trainable=True)
        self.conv2_b = tf.Variable(tf.constant(0.1, shape=[128]), trainable=True)
        self.conv3_w = tf.Variable(initializer(shape=[3, 3, 128, 128]), trainable=True)
        self.conv3_b = tf.Variable(tf.constant(0.1, shape=[128]), trainable=True)
        self.conv4_w = tf.Variable(initializer(shape=[3, 3, 128, 256]), trainable=True)
        self.conv4_b = tf.Variable(tf.constant(0.1, shape=[256]), trainable=True)
        self.conv5_w = tf.Variable(initializer(shape=[3, 3, 256, 256]), trainable=True)
        self.conv5_b = tf.Variable(tf.constant(0.1, shape=[256]), trainable=True)
        self.conv6_w = tf.Variable(initializer(shape=[3, 3, 256, 256]), trainable=True)
        self.conv6_b = tf.Variable(tf.constant(0.1, shape=[256]), trainable=True)
        self.conv7_w = tf.Variable(initializer(shape=[3, 3, 256, 512]), trainable=True)
        self.conv7_b = tf.Variable(tf.constant(0.1, shape=[512]), trainable=True)
        self.conv8_w = tf.Variable(initializer(shape=[3, 3, 512, 512]), trainable=True)
        self.conv8_b = tf.Variable(tf.constant(0.1, shape=[512]), trainable=True)
        self.conv9_w = tf.Variable(initializer(shape=[3, 3, 512, 512]), trainable=True)
        self.conv9_b = tf.Variable(tf.constant(0.1, shape=[512]), trainable=True)
        self.conv10_w = tf.Variable(initializer(shape=[3, 3, 512, 512]), trainable=True)
        self.conv10_b = tf.Variable(tf.constant(0.1, shape=[512]), trainable=True)
        self.conv11_w = tf.Variable(initializer(shape=[3, 3, 512, 512]), trainable=True)
        self.conv11_b = tf.Variable(tf.constant(0.1, shape=[512]), trainable=True)
        self.conv12_w = tf.Variable(initializer(shape=[3, 3, 512, 512]), trainable=True)
        self.conv12_b = tf.Variable(tf.constant(0.1, shape=[512]), trainable=True)
        self.fc1_w = tf.Variable(initializer(shape=[7 * 7 * 512, 4096]), trainable=True)
        self.fc1_b = tf.Variable(tf.constant(0.1, shape=[4096]), trainable=True)
        self.fc2_w = tf.Variable(initializer(shape=[4096, 4096]), trainable=True)
        self.fc2_b = tf.Variable(tf.constant(0.1, shape=[4096]), trainable=True)
        self.fc3_w = tf.Variable(initializer(shape=[4096, 5]), trainable=True)
        self.fc3_b = tf.Variable(tf.constant(0.1, shape=[5]), trainable=True)

    def __call__(self, train_data, training):
        # block 0
        conv0 = tf.nn.relu(tf.nn.conv2d(input=train_data, filters=self.conv0_w, strides=1, padding='SAME', name='conv0') + self.conv0_b)
        conv1 = tf.nn.relu(tf.nn.conv2d(input=conv0, filters=self.conv1_w, strides=1, padding='SAME', name='conv1') + self.conv1_b)
        max_pool0 = tf.nn.max_pool2d(input=conv1, ksize=[1, 2, 2, 1], strides=2, padding='SAME', name='max_pool0')

        # block 1
        conv2 = tf.nn.relu(tf.nn.conv2d(input=max_pool0, filters=self.conv2_w, strides=1, padding='SAME', name='conv2') + self.conv2_b)
        conv3 = tf.nn.relu(tf.nn.conv2d(input=conv2, filters=self.conv3_w, strides=1, padding='SAME', name='conv3') + self.conv3_b)
        max_pool1 = tf.nn.max_pool2d(input=conv3, ksize=[1, 2, 2, 1], strides=2, padding='SAME', name='max_pool1')

        # block 2
        conv4 = tf.nn.relu(tf.nn.conv2d(input=max_pool1, filters=self.conv4_w, strides=1, padding='SAME', name='conv4') + self.conv4_b)
        conv5 = tf.nn.relu(tf.nn.conv2d(input=conv4, filters=self.conv5_w, strides=1, padding='SAME', name='conv5') + self.conv5_b)
        conv6 = tf.nn.relu(tf.nn.conv2d(input=conv5, filters=self.conv6_w, strides=1, padding='SAME', name='conv6') + self.conv6_b)
        max_pool2 = tf.nn.max_pool2d(input=conv6, ksize=[1, 2, 2, 1], strides=2, padding='SAME', name='max_pool2')

        # block 3
        conv7 = tf.nn.relu(tf.nn.conv2d(input=max_pool2, filters=self.conv7_w, strides=1, padding='SAME', name='conv7') + self.conv7_b)
        conv8 = tf.nn.relu(tf.nn.conv2d(input=conv7, filters=self.conv8_w, strides=1, padding='SAME', name='conv8') + self.conv8_b)
        conv9 = tf.nn.relu(tf.nn.conv2d(input=conv8, filters=self.conv9_w, strides=1, padding='SAME', name='conv9') + self.conv9_b)
        max_pool3 = tf.nn.max_pool2d(input=conv9, ksize=[1, 2, 2, 1], strides=2, padding='SAME', name='max_pool3')

        # block 4
        conv10 = tf.nn.relu(tf.nn.conv2d(input=max_pool3, filters=self.conv10_w, strides=1, padding='SAME', name='conv10') + self.conv10_b)
        conv11 = tf.nn.relu(tf.nn.conv2d(input=conv10, filters=self.conv11_w, strides=1, padding='SAME', name='conv11') + self.conv11_b)
        conv12 = tf.nn.relu(tf.nn.conv2d(input=conv11, filters=self.conv12_w, strides=1, padding='SAME', name='conv12') + self.conv12_b)
        max_pool4 = tf.nn.max_pool2d(input=conv12, ksize=[1, 2, 2, 1], strides=2, padding='SAME', name='max_pool4')

        flat = tf.reshape(max_pool4, [-1, 7 * 7 * 512])
        fc1 = tf.nn.relu(tf.matmul(flat, self.fc1_w) + self.fc1_b)

        if training:
            fc1 = tf.nn.dropout(fc1, 0.5)

        fc2 = tf.nn.relu(tf.matmul(fc1, self.fc2_w) + self.fc2_b)

        if training:
            fc2 = tf.nn.dropout(fc2, 0.5)

        y_pred = tf.nn.softmax(tf.matmul(fc2, self.fc3_w) + self.fc3_b)

        return y_pred

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
    BATCH_SIZE = 50
    LR_INIT = 0.00001
    EPOCHS = 1
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    train_dataset, _, _ = get_dataset('/home/v100/tf_workspace/datasets/flower_set/train', True)
    test_dataset, _, _ = get_dataset('/home/v100/tf_workspace/datasets/flower_set/test', False)
    model = VGG16()

    optimizer = tf.keras.optimizers.SGD(learning_rate=LR_INIT)

    print()
    print('Learning started. It takes sometime.')
    for epoch in range(EPOCHS):
        avg_loss = 0.
        avg_train_acc = 0.
        avg_test_acc = 0.
        train_step = 0
        test_step = 0    
        
        for images, labels in train_dataset:
            train(model, images, labels)
            #grads = grad(model, images, labels)                
            #optimizer.apply_gradients(zip(grads, model.variables))
            loss = loss_fn(model, images, labels)
            acc = evaluate(model, images, labels)
            avg_loss = avg_loss + loss
            avg_train_acc = avg_train_acc + acc
            train_step += 1

        avg_loss = avg_loss / train_step
        avg_train_acc = avg_train_acc / train_step
        
        # print('Epoch:', '{}'.format(epoch + 1), 'loss =', '{:.8f}'.format(avg_loss), 
        #       'train accuracy = ', '{:.4f}'.format(avg_train_acc))

        for images, labels in test_dataset:        
            acc = evaluate(model, images, labels)        
            avg_test_acc = avg_test_acc + acc
            test_step += 1    
        avg_test_acc = avg_test_acc / test_step    

        print('Epoch:', '{}'.format(epoch + 1), 
              'loss =', '{:.8f}'.format(avg_loss), 
              'train accuracy = ', '{:.4f}'.format(avg_train_acc), 
              'test accuracy = ', '{:.4f}'.format(avg_test_acc))
        
    print('Learning Finished!')