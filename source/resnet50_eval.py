# -*- coding: utf-8 -*-
import tensorflow as tf
import json
from tensorflow.keras.applications.resnet50 import preprocess_input


project_name = '/home/barcelona/pervinco/model/four_shapes/2020.01.28_12:22' + '/'
test_data_dir = '/home/barcelona/pervinco/datasets/four_shapes/valid'
weight_path = '/home/barcelona/pervinco/model/four_shapes/2020.01.28_12:22/04-0.61.hdf5'
img_width, img_height, img_channel = 224, 224, 3
batch_size = 64
num_classes = 4

'''
custom model load
'''
# base_model = tf.keras.models.load_model('ALEX1_2class.h5')
# base_model.summary()
# base_model.outputs = [base_model.layers[-7].output]

'''
resnet50 model load
'''
# base_model = tf.keras.applications.ResNet50(include_top=False, input_shape=(img_width, img_height, img_channel),
#                                             weights='imagenet')
base_model = tf.keras.applications.ResNet50(include_top=False, input_shape=(img_width, img_height, img_channel),
                                            weights=None)
base_model.summary()
base_model.outputs = [base_model.layers[-1].output]

print(base_model.outputs)

last = base_model.outputs[0]
print(last)

x = tf.keras.layers.GlobalAveragePooling2D()(last)
preds = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

tl_model = tf.keras.Model(base_model.input, preds)
tl_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

for layer in base_model.layers:
    layer.trainalbe = True

tl_model.summary()

tl_model.load_weights(weight_path)

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
# test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=preprocess_input)


test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

tl_model.evaluate_generator(test_generator, steps=test_generator.n // batch_size, verbose=1)

model_json = tl_model.to_json()

with open(project_name + "CAM.json", "w") as json_file:
    json_file.write(model_json)

tl_model.save_weights(project_name + "CAM.h5")
