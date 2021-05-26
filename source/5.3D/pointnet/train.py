import numpy as np
import tensorflow as tf
import os, sys, h5py, datetime
from matplotlib import pyplot as plt
from transform_net import Input_Transform_Net, Feature_Transform_Net, ConvBN, FC

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


class ClsModel(tf.keras.Model):

    def __init__(self, num_points):
        super(ClsModel, self).__init__()
        self.num_points = num_points
        self.input_transform = Input_Transform_Net(self.num_points)
        self.feature_transform = Feature_Transform_Net(self.num_points, K=64)
        self.conv1 = ConvBN(64, [1, 3], padding='valid', strides=(1, 1), bn=True, name='conv1')
        # self.conv1 = tf.keras.layers.Conv2D(64, [1, 3], padding='valid', strides=(1, 1))
        self.conv2 = ConvBN(64, [1, 1], padding='valid', strides=(1, 1), bn=True, name='conv2')

        self.conv3 = ConvBN(64, [1, 1], padding='valid', strides=(1, 1), bn=True, name='conv3')
        self.conv4 = ConvBN(128, [1, 1], padding='valid', strides=(1, 1), bn=True, name='conv4')
        self.conv5 = ConvBN(1024, [1, 1], padding='valid', strides=(1, 1), bn=True, name='conv5')

        self.maxpooling = tf.keras.layers.MaxPool2D([self.num_points, 1], padding='valid')
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = FC(512, name='fc1')
        self.drop1 = tf.keras.layers.Dropout(0.5)
        self.fc2 = FC(256, name='fc2')
        self.drop2 = tf.keras.layers.Dropout(0.5)
        self.fc3 = FC(NUM_CLASSES, name='fc3', activation=False, bn=False)

    def call(self, inputs):
        # input_transform
        end_points = {}
        transform = self.input_transform(inputs)
        inputs = tf.matmul(inputs, transform)
        inputs= tf.expand_dims(inputs, -1)
        out = self.conv1(inputs)
        out = self.conv2(out)
        # feature_transform
        transform2 = self.feature_transform(out)
        end_points['transform'] = transform2
        out_transform = tf.matmul(tf.squeeze(out, axis=2), transform2)
        out_transform = tf.expand_dims(out_transform, axis=2)

        out_transform = self.conv3(out_transform)
        out_transform = self.conv4(out_transform)
        out_transform = self.conv5(out_transform)

        out_transform = self.maxpooling(out_transform)
        out_transform = self.flatten(out_transform)
        out_transform = self.fc1(out_transform)
        out_transform = self.drop1(out_transform)
        out_transform = self.fc2(out_transform)
        out_transform = self.drop2(out_transform)
        out_transform = self.fc3(out_transform)

        return tf.nn.softmax(out_transform)

    def active(self):
        x = tf.keras.Input(shape=(NUM_POINT, 3))

        return tf.keras.Model(inputs=[x], outputs=self.call(x))


def get_data_files(list_filename):
    return [line.rstrip() for line in open(list_filename)]


def load_h5(h5_filename):
    f = h5py.File(h5_filename, 'r')
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)


def shuffle_data(data, labels):
    idx = np.arange(len(labels))
    np.random.shuffle(idx)

    return data[idx, ...], labels[idx, :], idx


def rotate_point_cloud(batch_data):
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)

    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval], [0, 1, 0], [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)

    return rotated_data


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    B, N, C = batch_data.shape
    
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    jittered_data += batch_data

    return jittered_data
    

def preprocessing(x, y):
    x = tf.cast(x, dtype=tf.float32)
    y = tf.cast(y, dtype=tf.int64)
    y = tf.one_hot(y, depth=NUM_CLASSES)
    y = tf.squeeze(y, axis=0)

    return (x, y)


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


def display_training_curves(history):
    acc = history.history['categorical_accuracy']
    val_acc = history.history['val_categorical_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(len(history.history['loss']))

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    
    plt.savefig(f'/{SAVED_PATH}/train_result.png')
    plt.show()


def read_data(FILES_PATH, is_aug):
    total_pc = np.zeros(shape=[0, NUM_POINT, 3])
    total_label = np.zeros(shape=[0, 1])

    for i in range(0, len(FILES_PATH)):
        pc_data, pc_label = load_h5(TRAIN_FILES[i])
        pc_data = pc_data[:, 0:NUM_POINT, :]
        pc_data, pc_label, _ = shuffle_data(pc_data, pc_label)
        
        total_pc = np.append(total_pc, pc_data, axis=0)
        total_label = np.append(total_label, pc_label, axis=0)

    print(total_pc.shape, total_label.shape)

    if is_aug:
        rotated_pc = rotate_point_cloud(total_pc)
        rotated_label = total_label
        jitted_pc = jitter_point_cloud(total_pc)
        jitted_label = total_label

        total_pc = np.append(total_pc, rotated_pc, axis=0)
        total_pc = np.append(total_pc, jitted_pc, axis=0)
        total_label = np.append(total_label, rotated_label, axis=0)
        total_label = np.append(total_label, jitted_label, axis=0)

        print(total_pc.shape, total_label.shape)

    return total_pc, total_label


def train():
    total_pc, total_label = read_data(TRAIN_FILES, True)
    print(total_pc.shape, total_label.shape)

    TRAIN_STEPS_PER_EPOCH = int(tf.math.ceil(len(total_pc) / BATCH_SIZE).numpy())

    val_data, val_label = load_h5(VALID_FILES[0])
    val_data = val_data[:, 0:NUM_POINT, :]
    VALID_STEPS_PER_EPOCH = int(tf.math.ceil(len(val_data) / BATCH_SIZE).numpy())

    val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_label))
    val_dataset = val_dataset.map(preprocessing).batch(BATCH_SIZE).repeat()

    train_dataset = tf.data.Dataset.from_tensor_slices((total_pc, total_label))
    train_dataset = train_dataset.map(preprocessing).shuffle(10000).batch(BATCH_SIZE).repeat()
    
    model = ClsModel(NUM_POINT)
    model.active().summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE), loss=tf.keras.losses.categorical_crossentropy, metrics=['categorical_accuracy'])

    # model.load_weights(checkpoint_prefix)
    lrfn = build_lrfn()
    lr_schedule = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=1)
    earlystopper = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

    history = model.fit(train_dataset,
                        steps_per_epoch=TRAIN_STEPS_PER_EPOCH,
                        epochs=EPOCH,
                        validation_data=val_dataset,
                        validation_steps=VALID_STEPS_PER_EPOCH,
                        callbacks=[lr_schedule, earlystopper]
                        )

    tf.saved_model.save(model, f'{SAVED_PATH}/pointnet')

    return history


if __name__ == "__main__":
    BATCH_SIZE = 64
    NUM_CLASSES = 40
    NUM_POINT = 1024
    EPOCH = 250
    LEARNING_RATE = 0.0001
    MOMENTUM = 0.9
    DECAY_STEP = 200000
    DECAY_RATE = 0.7
    LOG_TIME = datetime.datetime.now().strftime("%Y.%m.%d_%H:%M")
    SAVED_PATH = f'/data/Models/pointnet/{LOG_TIME}'

    TRAIN_FILES = "/data/datasets/modelnet40_ply_hdf5_2048/train_files.txt"
    VALID_FILES = "/data/datasets/modelnet40_ply_hdf5_2048/test_files.txt"

    TRAIN_FILES = get_data_files(TRAIN_FILES)
    VALID_FILES = get_data_files(VALID_FILES)

    history = train()
    display_training_curves(history)