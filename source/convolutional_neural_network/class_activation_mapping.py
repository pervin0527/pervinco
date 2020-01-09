import tensorflow as tf

# img_width, img_height, img_channel = 224, 224, 3
# base_model = tf.keras.applications.ResNet50(include_top = False, input_shape=(img_width, img_height, img_channel))
# base_model.summary()
# nb_classes = 2
# base_model.outputs = [base_model.layers[-1].output]
# print(base_model.outputs)
#
# last = base_model.outputs[0]
# print(last)
# x = tf.keras.layers.GlobalAveragePooling2D()(last)

img_width, img_height, img_channel = 227, 227, 3

base_model = tf.keras.models.load_model('ALEX1_2class.h5')
base_model.summary()
base_model.outputs = [base_model.layers[-7].output]
print(base_model.outputs)

last = base_model.outputs[0]
print(last)

x = tf.keras.layers.GlobalAveragePooling2D()(last)
preds = tf.keras.layers.Dense(2, activation='softmax')(x)

tl_model = tf.keras.Model(base_model.input, preds)

for layer in base_model.layers:
    layer.trainalbe = False

tl_model.summary()

tl_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# es = EarlyStopping(patience=10, monitor='val_acc')
tb = tf.keras.callbacks.TensorBoard(log_dir='test')
filepath="weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
cp = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

train_data_dir = '/home/barcelona/pervinco/datasets/cats_and_dogs_small_set/train'
validation_data_dir = '/home/barcelona/pervinco/datasets/cats_and_dogs_small_set/validation'
test_data_dir = '/home/barcelona/pervinco/datasets/cats_and_dogs_small_set/validation'

datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                             horizontal_flip=True,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             shear_range=0.02)

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='categorical')


validation_generator = datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='categorical')

tl_model.fit_generator(generator=train_generator,
                       epochs=50,
                       steps_per_epoch=train_generator.n // 32,
                       validation_data=validation_generator,
                       validation_steps=validation_generator.n // 32,
                       callbacks=[cp, tb])