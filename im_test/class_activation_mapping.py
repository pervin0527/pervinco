import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications import resnet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Conv2D, ZeroPadding2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

img_width, img_height, img_channel = 224, 224, 3
batch_size = 32
epochs = 50
num_classes = 28
early_stop_patience = 5

# base_model = resnet50.ResNet50(include_top=False, input_shape=(img_width, img_height, img_channel), weights=None)
base_model = resnet50.ResNet50(include_top=False, input_shape=(img_width, img_height, img_channel), weights='imagenet')
base_model.outputs = [base_model.layers[-1].output]

last = base_model.outputs[0]
x = GlobalAveragePooling2D()(last)
preds = Dense(num_classes, activation='softmax')(x)

tl_model = Model(base_model.input, preds)

for layer in base_model.layers:
    # layer.trainable = False
    layer.trainable = True

tl_model.summary()
tl_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# optimizer = tf.keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# optimizer = tf.keras.optimizers.Adam(lr=0.1)#2e-5
# tl_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# es = EarlyStopping(patience=10, monitor='val_acc')
tb = TensorBoard(log_dir='/home/barcelona/pervinco/im_test/models/tensorboard/')
filepath="/home/barcelona/pervinco/im_test/models/weights/weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
cp = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
er = EarlyStopping(monitor='val_loss', patience=early_stop_patience)
rl = ReduceLROnPlateau(monitor='val_accuracy', factor=0.01, patience=3, mode='max', cool_down=1)

train_data_dir = '/home/barcelona/pervinco/im_test/datasets/train'
# train_data_dir = '/home/barcelona/pervinco/datasets/face_gender_glass/train'
validation_data_dir = '/home/barcelona/pervinco/im_test/datasets/test'
# validation_data_dir = '/home/barcelona/pervinco/datasets/face_gender_glass/validation'
# test_data_dir = '/home/barcelona/pervinco/im_test/datasets/test'

# datagen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=preprocess_input)
datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                                          horizontal_flip=True,
                                                          width_shift_range=0.2,
                                                          height_shift_range=0.2,
                                                          shear_range=0.02)

train_generator = datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')


validation_generator = datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

# test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
#
# test_generator = test_datagen.flow_from_directory(
#     test_data_dir,
#     target_size=(img_width, img_height),
#     batch_size=batch_size,
#     class_mode='categorical')


tl_model.fit_generator(generator=train_generator,
                       epochs=epochs,
                       steps_per_epoch=train_generator.n // batch_size,
                       validation_data=validation_generator,
                       validation_steps=validation_generator.n // batch_size,
                       callbacks=[tb, cp, rl])
