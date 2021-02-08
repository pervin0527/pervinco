import pathlib, cv2
import numpy as np
import tensorflow as tf
import albumentations as A

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

# GPU setup
gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 1:
    try:
        print("ActivateMulti GPU")
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


def get_dataset(path):
    dataset = pathlib.Path(path)

    image_files = list(dataset.glob('*.png'))
    image_files = [str(path) for path in image_files]

    images = []
    for path in image_files:
        image = cv2.imread(path)
        image = np.asarray(image, dtype='float32')
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        image = transform(image=image)['image']
        image = cv2.cvtColor(image ,cv2.COLOR_BGR2GRAY)
        image = image / 255.0
        image = np.reshape(image, (IMG_SIZE, IMG_SIZE, 1))
        images.append(image)

    train_images, valid_images = train_test_split(images, test_size=0.2)

    return np.asarray(train_images), np.asarray(valid_images)   


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


# class DenoisingAutoencoder(tf.keras.Model):
#   def __init__(self):
#     super(DenoisingAutoencoder, self).__init__()
#     self.encoder = tf.keras.Sequential([
#         tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)), 
#         tf.keras.layers.Conv2D(48, (3, 3), activation='relu', padding='same'),
#         tf.keras.layers.Conv2D(72, (3, 3), activation='relu', padding='same'),
#         tf.keras.layers.Conv2D(144, (3, 3), activation='relu', padding='same'),
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
#         tf.keras.layers.Dropout(0.5),
#     ])

#     self.decoder = tf.keras.Sequential([
#         tf.keras.layers.Conv2D(144, (3, 3), activation='relu', padding='same'),
#         tf.keras.layers.Conv2D(72, (3, 3), activation='relu', padding='same'),
#         tf.keras.layers.Conv2D(48, (3, 3), activation='relu', padding='same'),
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.UpSampling2D((2, 2)),
#         tf.keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')
#     ])

#   def call(self, x):
#     encoded = self.encoder(x)
#     decoded = self.decoder(encoded)
#     return decoded

def get_model():
    input_layer = tf.keras.Input(shape=(None, None, 1))
    
    e = tf.keras.layers.Conv2D(32, (3, 3), padding='same')(input_layer)
    e = tf.keras.layers.LeakyReLU(alpha=0.3)(e)
    e = tf.keras.layers.BatchNormalization()(e)
    e = tf.keras.layers.Conv2D(64, (3, 3), padding='same')(e)
    e = tf.keras.layers.LeakyReLU(alpha=0.3)(e)
    e = tf.keras.layers.BatchNormalization()(e)
    e = tf.keras.layers.Conv2D(64, (3, 3), padding='same')(e)
    e = tf.keras.layers.LeakyReLU(alpha=0.3)(e)
    e = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(e)

    d = tf.keras.layers.Conv2D(64, (3, 3), padding='same')(e)
    d = tf.keras.layers.LeakyReLU(alpha=0.3)(d)
    d = tf.keras.layers.BatchNormalization()(d)

    d = tf.keras.layers.Conv2D(64, (3, 3), padding='same')(d)
    d = tf.keras.layers.LeakyReLU(alpha=0.3)(d)
    d = tf.keras.layers.UpSampling2D((2, 2))(d)
    d = tf.keras.layers.Conv2D(32, (3, 3), padding='same')(d)
    d = tf.keras.layers.LeakyReLU(alpha=0.2)(d)
    output_layer = tf.keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(d)

    model = tf.keras.Model(input_layer, output_layer)

    return model


if __name__ == "__main__":
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    BATCH_SIZE = 16
    EPOCHS = 1000
    IMG_SIZE = 256

    original_set_path = '/data/backup/pervinco/datasets/dirty_mnist_2/map'
    noise_set_path = '/data/backup/pervinco/datasets/dirty_mnist_2/dirty_mnist_2nd'
    
    transform = A.Compose([A.Normalize(p=1),
                           A.RandomRotate90(p=0.4),
                           A.HorizontalFlip(p=0.4),
                           A.VerticalFlip(p=0.4)])

    train_set, valid_set = get_dataset(original_set_path)
    train_noise_set, valid_noise_set = get_dataset(noise_set_path)

    print(train_set.shape, train_noise_set.shape)
    print(valid_set.shape, valid_noise_set.shape)

    # early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=1)
    # autoencoder = DenoisingAutoencoder()
    # autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
    
    model = get_model()
    optimizer = tf.keras.optimizers.Adam(lr=9e-4, decay=1e-5)
    model.compile(loss='mse', optimizer=optimizer)
    model.summary()

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=1, mode='auto')

    history = model.fit(train_noise_set, train_set,
                              epochs=EPOCHS,
                              batch_size=BATCH_SIZE,
                              callbacks=[early_stopping],
                              validation_data=(valid_noise_set, valid_set))


    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.ylabel('LOSS')
    plt.xlabel('EPOCHS')
    plt.legend(['Train', 'Valid'], loc='upper left')
    plt.show()

    test_image = valid_noise_set[0]
    test_image = np.expand_dims(test_image, axis=0)
    print(test_image.shape)

    pred = model.predict(test_image)
    cv2.imshow('result', pred[0])
    cv2.waitKey(0)