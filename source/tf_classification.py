import sys
sys.path.append('../')
import json
import base64
import io
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
from datetime import datetime
import os.path
import random
import re
import time
from PIL import Image
import glob
import time
from keras.utils import np_utils
import tensorflow as tf
import keras
import cv2
import csv
from RedisQueue import RedisQueue
from  keras.backend.tensorflow_backend import set_session
from tensorflow.keras.applications.resnet50 import preprocess_input

ssd_q = RedisQueue("ssd", port=6480)
final_q = RedisQueue('', port=6483)


with open('./cu50_mapping.csv', 'r') as df:
    reader = csv.reader(df)
    CLASS_NAMES = list(reader)
    

def init_model(test_img):
    img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)
    
    return img


def set_gpu_option(which_gpu, fraction_memory):
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = fraction_memory
    config.gpu_options.visible_device_list = which_gpu
    set_session(tf.Session(config=config))
    return
    
set_gpu_option("0", 0.7)

if __name__ == '__main__':
    model = tf.keras.models.load_model('/home/test/500_hand/tf_model/cu50.h5')
    test = init_model(np.zeros((224, 224, 3), np.uint8))
    model.predict(test, steps=1)
    

    while True:
        if ssd_q.qsize == 0:
            time.sleep(0.02)
            pass

        else:
            t_start_time = time.time()
            i_received_data = ssd_q.get()
            i_unpacked_data = json.loads(str(i_received_data))
            i_key = i_unpacked_data.keys()[0]
            i_result = i_unpacked_data[i_key]
            i_frame = []

            for i in range(0, len(i_result)):
                a_frame = []
                i_img = i_result[i]["image"]
                i_num = i_result[i]["num"]
                i_device_id = i_result[i]["device_id"]
                i_x = i_result[i]["x"]
                i_y = i_result[i]["y"]
                i_w = i_result[i]["w"]
                i_h = i_result[i]["h"]
                i_timestamp = i_result[i]["timestamp"]

                inputloc = base64.b64decode(i_img)
                ##inputloc = Image.open(io.BytesIO(inputloc)).convert('RGB')
                inputloc = Image.open(io.BytesIO(inputloc))
                #inputloc = io.BytesIO(inputloc)
                inputloc = inputloc.resize((224, 224))
                inputloc = np.array(inputloc)

                inputloc = np.expand_dims(inputloc, axis=0)
                inputloc = preprocess_input(inputloc)
                ##inputloc = data_generator.flow(inputloc)

                predictions = model.predict(inputloc, steps=1)
                score = np.argmax(predictions[0])

                #barcode = str(CLASS_NAMES[score])
                barcode = CLASS_NAMES[score][1]
                #print(barcode)
                
                a_frame.append({"device_id":i_device_id, "x":i_x, "y":i_y, "w":i_w, "h":i_h, "barcode":barcode, "num":i_num, "timestamp":i_timestamp})
                i_dict = {i_key : a_frame}
                i_json_keys = json.dumps(i_dict)                   
                final_q.put(i_json_keys, key='final_' + i_key)
                print('-----%s seconds----' %(time.time()-t_start_time))
                
            #print(i_frame)


