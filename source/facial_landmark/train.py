import os
import tensorflow as tf
from PFLD import PFLD

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