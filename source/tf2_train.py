import tensorflow as tf
import glob
import os
import datetime
from tensorflow.keras.applications.resnet50 import preprocess_input

DATASET_NAME = 'cu50'
MODEL_NAME = DATASET_NAME

train_dir = '/home/barcelona/pervinco/datasets/' + DATASET_NAME + '/train3'
valid_dir = '/home/barcelona/pervinco/datasets/' + DATASET_NAME + '/valid3'
NUM_CLASSES = len(glob.glob(train_dir + '/*'))

CHANNELS = 3
IMAGE_RESIZE = 224
NUM_EPOCHS = 10
BATCH_SIZE = 32
EARLY_STOP_PATIENCE = 5

saved_path = '/home/barcelona/pervinco/model/'
time = datetime.datetime.now().strftime("%Y.%m.%d_%H:%M") + '_keras2'
weight_file_name = '{epoch:02d}-{val_accuracy:.2f}.hdf5'

if not (os.path.isdir(saved_path + DATASET_NAME + '/' + time)):
    os.makedirs(os.path.join(saved_path + DATASET_NAME + '/' + time))
else:
    pass

resnet_weights_path = '/home/barcelona/pervinco/source/weights/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

model = tf.keras.models.Sequential()
model.add(tf.keras.applications.ResNet50(include_top=False, pooling='avg', weights=None))
model.add(tf.keras.layers.Dense(NUM_CLASSES, activation='softmax'))
model.layers[0].trainable = True
model.summary()

optimizer = tf.keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

data_generator = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = data_generator.flow_from_directory(train_dir,
                                                     target_size=(IMAGE_RESIZE, IMAGE_RESIZE),
                                                     batch_size=BATCH_SIZE,
                                                     class_mode='categorical')

valid_generator = data_generator.flow_from_directory(valid_dir,
                                                     target_size=(IMAGE_RESIZE, IMAGE_RESIZE),
                                                     batch_size=BATCH_SIZE,
                                                     class_mode='categorical')

# print(train_generator[0])


cb_early_stopper = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=EARLY_STOP_PATIENCE)
cb_checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath=saved_path + DATASET_NAME + '/' +
                                                              time + '/' + weight_file_name,
                                                     monitor='val_accuracy', save_best_only=True, mode='auto')

fit_history = model.fit(train_generator,
                        steps_per_epoch=train_generator.n / BATCH_SIZE,
                        epochs=NUM_EPOCHS,
                        shuffle=False,
                        validation_data=valid_generator,
                        validation_steps=valid_generator.n / BATCH_SIZE,
                        callbacks=[cb_early_stopper, cb_checkpointer])

model.save(saved_path + DATASET_NAME + '/' + time + '/' + MODEL_NAME + '.h5')
