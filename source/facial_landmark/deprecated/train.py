import os
import tensorflow as tf
from IPython.display import clear_output

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


# def plot_predictions(dataset, model):
#     for item in dataset.take(1):
#         image = item[0][0].numpy()
#         image = tf.expand_dims(image, axis=0)
#         prediction = model.predict(image, verbose=0)
#         landmarks = prediction[0].reshape(98, 2)
#         print(landmarks.shape)        


# class DisplayCallback(tf.keras.callbacks.Callback):
#     def on_epoch_end(self, epoch, logs=None):
#         clear_output(wait=True)
#         plot_predictions(test_dataset, model=model)


def inverted_residual_block(input, expansion, stride, filters, use_res_connect):
    input_channel = tf.keras.backend.int_shape(input)[-1]
    assert stride in [1, 2]

    block = tf.keras.Sequential([
    tf.keras.layers.Conv2D(expansion * input_channel, kernel_size=1, padding="valid"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Conv2D(expansion * input_channel, kernel_size=3, strides=stride, padding="same"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Conv2D(filters=filters, kernel_size=1, strides=1, padding="valid"),
    tf.keras.layers.BatchNormalization(),
    ])

    if use_res_connect:
        # return tf.keras.layers.Add()([input, x])
        return input + block(input)

    return block(input)


def backbone(input_size, n_landmarks):
    input_layer = tf.keras.layers.Input(shape=(input_size, input_size, 3))
    
    ## 112, 112, 3 ----> 56, 56, 64
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, padding="same")(input_layer)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    ## 56, 56, 64 ----> 56, 56, 64
    # x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, padding="same")(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    ## 56, 56, 64 ----> 28, 28, 64
    ## repeated 5, stride first layer : 2, expansion : 2
    x = inverted_residual_block(x, expansion=2, stride=2, filters=64, use_res_connect=False)
    x = inverted_residual_block(x, expansion=2, stride=1, filters=64, use_res_connect=True)
    x = inverted_residual_block(x, expansion=2, stride=1, filters=64, use_res_connect=True)
    x = inverted_residual_block(x, expansion=2, stride=1, filters=64, use_res_connect=True)
    out1 = inverted_residual_block(x, expansion=2, stride=1, filters=64, use_res_connect=True) ## 28, 28, 64

    ## 28, 28, 64 ----> 14, 14, 128
    x = inverted_residual_block(out1, expansion=2, stride=2, filters=128, use_res_connect=False)

    ## 14, 14, 128 ----> 14, 14, 128
    x = inverted_residual_block(x, expansion=4, stride=1, filters=128, use_res_connect=False)
    x = inverted_residual_block(x, expansion=4, stride=1, filters=128, use_res_connect=True)
    x = inverted_residual_block(x, expansion=4, stride=1, filters=128, use_res_connect=True)
    x = inverted_residual_block(x, expansion=4, stride=1, filters=128, use_res_connect=True)
    x = inverted_residual_block(x, expansion=4, stride=1, filters=128, use_res_connect=True)
    x = inverted_residual_block(x, expansion=4, stride=1, filters=128, use_res_connect=True)

    ## 14, 14, 128 ----> 14, 14, 16
    x = inverted_residual_block(x, expansion=2, stride=1, filters=16, use_res_connect=False)

    ## S1 : 14, 14, 16
    s1 = tf.keras.layers.AvgPool2D(14)(x)
    s1 = tf.keras.layers.Flatten()(s1)
    
    ## S2 : 14, 14, 16 ----> 7, 7, 32
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=2, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    s2 = tf.keras.layers.AvgPool2D(7)(x)
    s2 = tf.keras.layers.Flatten()(s2)

    ## S3 : 7, 7, 32 ----> 1, 1, 128
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=7, strides=1, padding="valid")(x)
    x = tf.keras.layers.ReLU()(x)

    s3 = tf.keras.layers.Flatten()(x)

    multi_scale = tf.keras.layers.concatenate([s1, s2, s3], axis=1)
    output_layer = tf.keras.layers.Dense(n_landmarks)(multi_scale)

    model = tf.keras.Model(inputs=input_layer, outputs=[out1, output_layer])
    return model


def auxiliarynet(input_size):
    input_layer = tf.keras.layers.Input(shape=(input_size, input_size, 64)) ## 28, 28, 64

    ## 28, 28, 64 ---> 14, 14, 128
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=2, padding="same")(input_layer)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    ## 14, 14, 128 ---> 14, 14, 128
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    ## 14, 14, 128 ---> 7, 7, 32
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=2, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    ## 7, 7, 32 ----> 3, 3, 128
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=7, strides=3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.MaxPool2D(pool_size=3, padding="same")(x) ## 1, 1, 128
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(32)(x)
    output_layer = tf.keras.layers.Dense(3)(x)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    return model


def novel_loss(attribute_gt, landmark_gt, euler_angle_gt, angle, landmarks):
    weight_angle = tf.reduce_sum(1 - tf.cos(angle - euler_angle_gt), axis=1)
    attributes_w_n = attribute_gt[:, 1:6]
    # attributes_w_n = tf.where(attributes_w_n > 0, attributes_w_n, 0.1 / cfg.BATCH_SIZE)

    mat_ratio = tf.reduce_mean(attributes_w_n, axis=0)
    mat_ratio = tf.where(mat_ratio > 0, 1.0 / mat_ratio, BATCH_SIZE)

    weight_attribute = tf.reduce_sum(tf.multiply(attributes_w_n, mat_ratio), axis=1)

    l2_distance = tf.reduce_sum((landmark_gt - landmarks) * (landmark_gt - landmarks), axis=1)

    weighted_loss = tf.reduce_mean(weight_angle * weight_attribute * l2_distance)

    loss = tf.reduce_mean(l2_distance)

    return weighted_loss, loss


class PFLD(tf.keras.Model):
    def __init__(self, input_size=112, n_landmarks=98 * 2, summary=False):
        super(PFLD, self).__init__()
        self.pfld_model = backbone(input_size, n_landmarks)
        self.auxiliary_model = auxiliarynet(input_size=int(input_size / 4))

        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.loss_tracker_2 = tf.keras.metrics.Mean(name="weighted_loss")

        if summary:
            self.pfld_model.summary()
            self.auxiliary_model.summary()

    def call(self, x, training=False):
        features, landmark = self.pfld_model(x)
        if training:
            angle = self.auxiliary_model(features)
            return angle, landmark
        else:
            return landmark

    def train_step(self, data):
        img_tensor, attribute_gt, landmark_gt, euler_angle_gt = data
        
        with tf.GradientTape() as tape:
            angle, landmarks = self(img_tensor, training=True)
            weighted_loss, loss = novel_loss(attribute_gt, landmark_gt, euler_angle_gt, angle, landmarks)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(weighted_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.loss_tracker.update_state(loss)
        self.loss_tracker_2.update_state(weighted_loss)
        
        return {"loss": self.loss_tracker.result(), "weighted_loss": self.loss_tracker_2.result()}

    @property
    def metrics(self):
        return [self.loss_tracker, self.loss_tracker_2]


def data_process(data):
    splits = tf.strings.split(data, sep=' ')
    image_path = splits[0]
    image_file = tf.io.read_file(image_path)
    image = tf.io.decode_jpeg(image_file, channels=3)
    image.set_shape([IMG_SIZE, IMG_SIZE, 3])

    landmarks = splits[1:197]
    landmarks = tf.strings.to_number(landmarks, out_type=tf.float32)
    landmarks.set_shape([N_LANDMARKS])
    
    attribute = splits[197:203]
    attribute = tf.strings.to_number(attribute, out_type=tf.float32)
    attribute.set_shape([6])
    
    euler_angle = splits[203:206]
    euler_angle = tf.strings.to_number(euler_angle, out_type=tf.float32)
    euler_angle.set_shape([3])

    return image, attribute, landmarks, euler_angle

    # feature_description = {"image":image, "attribute":attribute, "landmark":landmarks, "euler_angle":euler_angle}
    # return feature_description


def make_tf_data(txt_file):
    dataset = tf.data.TextLineDataset(txt_file)
    dataset = dataset.map(data_process, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size=BATCH_SIZE)
    # dataset = dataset.repeat()
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset


if __name__ == "__main__":
    EPOCHS = 10
    IMG_SIZE = 112
    BATCH_SIZE = 64
    N_LANDMARKS = 98 * 2
    LR = 0.00001

    train_file_list = "/data/Source/PFLD-pytorch/data/train_data/list.txt"
    test_file_list = "/data/Source/PFLD-pytorch/data/test_data/list.txt"
    
    train_dataset = make_tf_data(train_file_list)
    test_dataset = make_tf_data(test_file_list)

    print(train_dataset)
    print(test_dataset)

    optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
    model = PFLD(input_size=IMG_SIZE, n_landmarks=N_LANDMARKS, summary=True)
    model.compile(optimizer=optimizer)

    TRAIN_STEPS_PER_EPOCH = int(tf.math.ceil(75000 / BATCH_SIZE).numpy())
    TEST_STEPS_PER_EPOCH = int(tf.math.ceil(2500 / BATCH_SIZE).numpy())

    history = model.fit(
        train_dataset,
        # steps_per_epoch=TRAIN_STEPS_PER_EPOCH,
        validation_data=test_dataset,
        # validation_steps=TEST_STEPS_PER_EPOCH,
        verbose=1,
        epochs=EPOCHS
    )