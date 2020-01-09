import tensorflow as tf
import json

test_data_dir = '/home/barcelona/pervinco/datasets/cats_and_dogs_small_set/validation'
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
tl_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

for layer in base_model.layers:
    layer.trainalbe = False

tl_model.summary()

tl_model.load_weights('./weights/weights-improvement-39-0.90.hdf5')

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='categorical')

tl_model.evaluate_generator(test_generator, steps=test_generator.n // 32)

model_json = tl_model.to_json()

with open("resnet_CAM_new.json", "w") as json_file:
    json_file.write(model_json)

tl_model.save_weights("alexnet_CAM_new.h5")
