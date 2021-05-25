import datetime
import pandas as pd
import numpy as np
import tensorflow as tf
from tqdm import tqdm
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


def rotate_point_cloud(points):
    rotated_data = np.zeros(points.shape, dtype=np.float32)

    for k in range(points.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval], [0, 1, 0], [-sinval, 0, cosval]])
        shape_pc = points[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)

    return rotated_data


def jitter_point_cloud(points, sigma=0.01, clip=0.05):
    B, N, C = points.shape

    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    jittered_data += points

    return jittered_data


def read_data_list(file_path):
    files = pd.read_csv(file_path, sep=' ', index_col=False, header=None)
    files = sorted(files[0].tolist())

    points = np.zeros(shape=[0, NUM_POINT, 3])
    labels = np.zeros(shape=[0, 1])

    for i in tqdm(range(len(files))):
        label = files[i].split('_')
        label = label[:-1]
        label = '_'.join(label)
        path = f'{DATA_PATH}/{label}/{files[i]}.txt'
        point = pd.read_csv(path, sep=',', index_col=False, header=None, names=['x', 'y', 'z', 'r', 'g', 'b'])

        point = point.loc[:,['x','y','z']]
        point = np.array(point)
        point = point[0:NUM_POINT, :]
        # print(point.shape)

        label = np.array([CLASSES.index(label)])
        # print(point.shape, label.shape)

        points = np.append(points, [point], axis=0)
        labels = np.append(labels, [label], axis=0)

    # print(points.shape, labels.shape)

    rotated_points = rotate_point_cloud(points)
    jitted_points = jitter_point_cloud(points)

    total_points = np.append(points, rotated_points, axis=0)
    total_points = np.append(total_points, jitted_points, axis=0)
    total_labels = np.append(labels, labels, axis=0)
    total_labels = np.append(total_labels, labels, axis=0)

    print(total_points.shape, total_labels.shape)

    return total_points, total_labels


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


def preprocessing(x, y):
    x = tf.cast(x, dtype=tf.float32)
    y = tf.cast(y, dtype=tf.int64)
    y = tf.one_hot(y, depth=NUM_CLASSES)
    y = tf.squeeze(y, axis=0)
    return (x, y)


def train(train_points, train_labels, valid_points, valid_labels):
    TRAIN_STEPS_PER_EPOCH = int(tf.math.ceil(len(train_labels) / BATCH_SIZE).numpy())
    VALID_STEPS_PER_EPOCH = int(tf.math.ceil(len(valid_labels) / BATCH_SIZE).numpy())

    train_dataset = tf.data.Dataset.from_tensor_slices((train_points, train_labels))
    train_dataset = train_dataset.map(preprocessing).shuffle(10000).batch(BATCH_SIZE).repeat()

    val_dataset = tf.data.Dataset.from_tensor_slices((valid_points, valid_labels))
    val_dataset = val_dataset.map(preprocessing).batch(BATCH_SIZE).repeat()
    
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
    DATA_PATH = '/data/datasets/modelnet40_normal_resampled'
    BATCH_SIZE = 64
    NUM_CLASSES = 40
    NUM_POINT = 8192
    EPOCH = 250
    LEARNING_RATE = 0.0001
    MOMENTUM = 0.9
    DECAY_STEP = 200000
    DECAY_RATE = 0.7
    LOG_TIME = datetime.datetime.now().strftime("%Y.%m.%d_%H:%M")
    SAVED_PATH = f'/data/Models/pointnet/{LOG_TIME}'

    CLASS_FILE = f'{DATA_PATH}/modelnet40_shape_names.txt'
    CLASS_FILE = pd.read_csv(CLASS_FILE, sep=' ', index_col=False, header=None)
    CLASSES = sorted(CLASS_FILE[0].tolist())
    print(CLASSES)

    TRAIN_FILE = f'{DATA_PATH}/modelnet40_train.txt'
    VALID_FILE = f'{DATA_PATH}/modelnet40_test.txt'

    train_points, train_labels = read_data_list(TRAIN_FILE)
    valid_points, valid_labels = read_data_list(VALID_FILE)

    train(train_points, train_labels, valid_points, valid_labels)