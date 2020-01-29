import tensorflow as tf
import cv2
import glob
import numpy as np

model_path = '/home/barcelona/pervinco/source/weights/product11.h5'
img_path = '/home/barcelona/pervinco/source/etc/cam_num_0_1579498371.84_crop.jpg'
dataset_path = glob.glob('/home/barcelona/nb_classification/datasets/original/original_raw/*')
class_list = []

for l in dataset_path:
    label = l.split('/')[-1]
    class_list.append(label)

class_list = sorted(class_list)

img_resize = 224

if __name__ == '__main__':
    model = tf.keras.models.load_model(model_path)
    img = cv2.imread(img_path)
    img = cv2.resize(img , (img_resize, img_resize))
    img_arr = img / 2
    img_tensor = np.expand_dims(img_arr, 0)
    
    predictions = model.predict(img_tensor)
    print('label number : ', np.argmax(predictions[0]), 'label name : ', class_list[np.argmax(predictions[0])])

