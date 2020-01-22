# -*- coding: utf-8 -*-
import tensorflow as tf
import datetime
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications import resnet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Conv2D, ZeroPadding2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

'''
기본 설정값 및 데이터 경로
'''
img_width, img_height, img_channel = 299, 299, 3
batch_size = 32
epochs = 50
num_classes = 2
early_stop_patience = 5
log_dir = '/home/barcelona/pervinco/model/cats_and_dogs'
time = datetime.datetime.now().strftime("%Y.%m.%d_%H:%M")
filepath = log_dir + '/' + str(time) + '/{epoch:02d}-{val_accuracy:.2f}.hdf5'

train_data_dir = '/home/barcelona/pervinco/datasets/cats_and_dogs_small_set/train'
validation_data_dir = '/home/barcelona/pervinco/datasets/cats_and_dogs_small_set/validation'
# test_data_dir = '/home/barcelona/pervinco/datasets/chest_xray/test'

'''
모델 선언
'''
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

'''
콜백 값 설정
TensorBoard - 텐서 보드 로그 기록
ModelCheckpoint - weight 파일 저장, val acc이 오를 때만 저장.
EarlyStopping - patience 만큼 val loss가 감소하지 않을 경우 training이 멈춤.
'''
# es = EarlyStopping(patience=10, monitor='val_acc')
tb = TensorBoard(log_dir + '/' + str(time) + '/tb')
cp = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
er = EarlyStopping(monitor='val_loss', patience=early_stop_patience)
# rl = ReduceLROnPlateau(monitor='val_accuracy', factor=0.01, patience=3, mode='max', cool_down=1)

'''
image augmentation
'''
datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
# datagen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=preprocess_input)
# datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
#                                                           rotation_range=10,
#                                                           horizontal_flip=True,
#                                                           vertical_flip=True,
#                                                           width_shift_range=0.2,
#                                                           height_shift_range=0.2)

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)


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


# test_generator = test_datagen.flow_from_directory(
#     test_data_dir,
#     target_size=(img_width, img_height),
#     batch_size=batch_size,
#     class_mode='categorical')

'''
training 시작
'''
tl_model.fit_generator(generator=train_generator,
                       epochs=epochs,
                       steps_per_epoch=train_generator.n // batch_size,
                       validation_data=validation_generator,
                       validation_steps=validation_generator.n // batch_size,
                       callbacks=[tb, cp])


