import random
import pathlib
import numpy as np
import tensorflow as tf
import albumentations as A


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


def get_file_list(ds_path, is_train):
    ds_path = pathlib.Path(ds_path)

    images = list(ds_path.glob('*/*'))
    images = [str(path) for path in images]
    
    classes = sorted(item.name for item in ds_path.glob('*/') if item.is_dir())
    labels = dict((name, index) for index, name in enumerate(classes))
    labels = [labels[pathlib.Path(path).parent.name] for path in images]

    if is_train:
        return images, labels, classes
    else:
        return images, labels


def generator(x, y, batch_size):
    while True:
        random.shuffle(x)
        random.shuffle(y)

        for offset in range(0, len(x), batch_size):
            X = []
            Y = []

            for image, label in zip(x, y):
                image = tf.keras.preprocessing.image.load_img(image, color_mode='rgb', target_size=(224, 224))
                image = tf.keras.preprocessing.image.img_to_array(image)
                # image = tf.expand_dims(image, 0)

                label = tf.one_hot(label, len(classes))
                
                X.append(image)
                Y.append(label)

            X = np.array(X)
            Y = np.array(Y)
            print(X.shape, Y.shape)
            yield X, Y


def get_model(classes):
    with strategy.scope():
        base_model = tf.keras.applications.EfficientNetB0(input_shape=(224, 224, 3),
                                                          weights="imagenet", # noisy-student
                                                          include_top=False)
        for layer in base_model.layers:
            layer.trainable = True
            
        avg = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
        output = tf.keras.layers.Dense(len(classes), activation="softmax")(avg)
        model = tf.keras.Model(inputs=base_model.input, outputs=output)

    model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics = ['categorical_accuracy'])
    model.summary()
    return model

                
if __name__ == "__main__":
    train_path = "/home/ubuntu/Datasets/Seeds/natural_image/train"
    valid_path = "/home/ubuntu/Datasets/Seeds/natural_image/valid"
    batch_size = 1
    epoch = 1

    train_x, train_y, classes = get_file_list(train_path, True)
    valid_x, valid_y = get_file_list(valid_path, False)
    print(classes)

    train_generator = generator(train_x, train_y, batch_size)
    valid_generator = generator(valid_x, valid_y, batch_size)
    
    model = get_model(classes)
    model.fit(train_generator,
              validation_data=valid_generator,
              epochs=epoch,
              steps_per_epoch=int(len(train_x) / batch_size),
              validation_steps=int(len(valid_x) / batch_size))