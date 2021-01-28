import cv2, random, pathlib, os
import numpy as np
import tensorflow as tf

def basic_processing(ds_path, is_training):
    ds_path = pathlib.Path(ds_path)

    images = list(ds_path.glob('*/*'))
    images = [str(path) for path in images]

    if is_training:
        random.shuffle(images)

    labels = sorted(item.name for item in ds_path.glob('*/') if item.is_dir())
    num_of_labels = len(labels)
    labels = dict((name, index) for index, name in enumerate(labels))
    labels = [labels[pathlib.Path(path).parent.name] for path in images]
    # labels = tf.keras.utils.to_categorical(labels, num_classes=num_of_labels, dtype='float32')

    return images, labels, num_of_labels


def cutmix(image, label, PROBABILITY = 1.0):
    DIM = IMAGE_SIZE[0]
    
    imgs = []; labs = []
    for j in range(BATCH_SIZE):
        print("#########################################################################")
        P = tf.cast(tf.random.uniform([], 0, 1) <= PROBABILITY, tf.int32) # 0 ~ 1 사이 난수 생성후, PROB보다 이하면 1, 초과면 0
        k = tf.cast(tf.random.uniform([], 0, BATCH_SIZE), tf.int32) # BATCH_SIZE 보다 작은 난수 생성.
        print(label[j], label[k])
        x = tf.cast(tf.random.uniform([], 0, DIM), tf.int32) # 0 ~ IMAGE_SIZE로 난수 생성
        y = tf.cast(tf.random.uniform([], 0, DIM), tf.int32)
        b = tf.random.uniform([], 0, 1) # 0 ~ 1사이 난수 생성.
        print(P, k, x, y, b)

        WIDTH = tf.cast(DIM * tf.math.sqrt(1 - b), tf.int32) * P
        print(WIDTH)

        ya = tf.math.maximum(0, y - WIDTH // 2)
        yb = tf.math.minimum(DIM, y + WIDTH // 2)
        xa = tf.math.maximum(0, x - WIDTH // 2)
        xb = tf.math.minimum(DIM, x + WIDTH // 2)
        print(ya, yb, xa, xb)
        
        one = image[j, ya : yb, 0 : xa, :]
        two = image[k, ya : yb, xa : xb, :]
        three = image[j, ya : yb, xb : DIM, :]
        middle = tf.concat([one, two, three], axis=1)
        img = tf.concat([image[j, 0 : ya, : , :], middle, image[j, yb : DIM, :, :]], axis=0)
        
        a = tf.cast(WIDTH * WIDTH / DIM / DIM, tf.float32)
        print(a)
        
        lab1 = tf.one_hot(label[j], CLASSES)
        lab2 = tf.one_hot(label[k], CLASSES)
        print(lab1, lab2)

        print((1 - a) * lab1 + a * lab2)
        cv2.imshow('result', np.uint8(img))
        cv2.waitKey(0)
        # break


def decode_image(image_data):
    image = tf.io.read_file(image_data)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    # image = tf.cast(image, tf.float32) / 255.0
    image = tf.keras.applications.efficientnet.preprocess_input(image)
    image = tf.reshape(image, [*IMAGE_SIZE, 3])
    return image


def onehot_encoding(label):
    return tf.one_hot(label, CLASSES)


def images_to_arr(images, labels):
    X = np.zeros([len(images), 224, 224, 3], dtype=np.uint8)
    y = np.zeros([len(labels), 1], dtype=np.uint8)

    for i in range(len(images)):
        image = images[i]
        image = cv2.imread(image)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        X[i] = image
        y[i] = labels[i]

    return X, y


if __name__ == "__main__":
    BATCH_SIZE = 16
    IMAGE_SIZE = [224, 224]
    AUTO = tf.data.experimental.AUTOTUNE
    DATASET_PATH = '/data/backup/pervinco/datasets/test'

    images, labels, CLASSES = basic_processing(DATASET_PATH, True)
    print(len(images), len(labels))
    images_arr, labels_arr = images_to_arr(images, labels)
    print(labels_arr)
    cutmix(images_arr, labels_arr)