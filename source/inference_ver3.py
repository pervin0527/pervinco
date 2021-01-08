import tensorflow as tf
import glob, os, sys, time, cv2, argparse
import queue
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from xml.dom import minidom
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

os.environ["CUDA_VISIBLE_DEVICES"]="0"

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        print("True")
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   try:
#     tf.config.experimental.set_virtual_device_configuration(gpus[0],
#       [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=9000)])
#   except RuntimeError as e:
#     print(e)


def get_boxes(label_path):
    xml_path = os.path.join(label_path)
    root_1 = minidom.parse(xml_path)
    bnd_1 = root_1.getElementsByTagName('bndbox')
    result = []
    for i in range(len(bnd_1)):
        xmin = int(bnd_1[i].childNodes[1].childNodes[0].nodeValue)
        ymin = int(bnd_1[i].childNodes[3].childNodes[0].nodeValue)
        xmax = int(bnd_1[i].childNodes[5].childNodes[0].nodeValue)
        ymax = int(bnd_1[i].childNodes[7].childNodes[0].nodeValue)
        result.append((xmin,ymin,xmax,ymax))
    return result


def img_preprocess(main_boxes, empty_boxes, img):
    resize = (224, 224)
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for index, boxes in enumerate([main_boxes, empty_boxes]):
        each_column = list(map(lambda j : image[j[1]:j[3],j[0]:j[2]], boxes))
        try:
            # main
            if index == 0:
                main_images = list(map(lambda j : cv2.resize(j, resize), each_column))
                main_images = list(map(lambda j : preprocess_input(j), main_images))
            # empty
            else:
                empty_images = list(map(lambda j : cv2.resize(j, resize), each_column))
                empty_images = list(map(lambda j : preprocess_input(j), empty_images))
        except Exception as e:
            print(e)
            
    return main_images, empty_images


class InferenceClass:
    def __init__(self, empty_model_path, empty_label_path, main_model_path, main_label_path):

        self.empty_model = tf.keras.models.load_model(empty_model_path)
        self.empty_df = pd.read_csv(empty_label_path, sep= ' ', index_col = False, header=None)
        self.empty_class_names = sorted(self.empty_df[0].tolist())

        self.main_model = tf.keras.models.load_model(main_model_path)
        self.main_df = pd.read_csv(main_label_path, sep= ' ', index_col = False, header=None)
        self.main_class_names = sorted(self.main_df[0].tolist())

    def predict(self, data, mode):
        if mode == 'empty':
            predictions = self.empty_model.predict(data)
            score = np.argmax(predictions, axis=1)
            return list(map(lambda x: self.empty_class_names[x], score))
        else:
            predictions = self.main_model.predict(data)
            score = np.argmax(predictions, axis=1)
            return list(map(lambda x: self.main_class_names[x], score))


def save_images(left_main_boxes, right_main_boxes, left_frame, right_frame):
    now = datetime.now().strftime("%Y.%m.%d_%H:%M:%S")
    save_path = f'./inference/logs/{store_name}'

    if os.path.isdir(save_path):
        pass
    else:
        os.makedirs(save_path+'/frame')
        os.makedirs(save_path+'/crop')

    cv2.imwrite(save_path + '/frame/left' + now + '.jpg', left_frame)
    cv2.imwrite(save_path + '/frame/right' + now + '.jpg', right_frame)

    for idx, (xmin, ymin, xmax, ymax) in enumerate(left_main_boxes):
        left_crop = left_frame[ymin:ymax, xmin:xmax]
        cv2.imwrite(save_path + '/crop/left_' + str(idx) + now + '.jpg', left_crop)

    for idx, (xmin, ymin, xmax, ymax) in enumerate(right_main_boxes):
        right_crop = right_frame[ymin:ymax, xmin:xmax]
        cv2.imwrite(save_path +'/crop/right_' + str(idx) + now + '.jpg', right_crop)

    print("Frame {} saved in {}".format(now, save_path))
    time.sleep(0.5)


def inference_test_images():
    os.system('clear')

    empty_final = []
    main_final = []
    spliter = []

    folder = store_name.split('_')[0]
    device_id = os.listdir(f'./inference/test_sets/{folder}/')
    print(device_id)

    if len(device_id) > 1:
        device_id = str(input("choose device_id : "))

    else:
        device_id = device_id[0]

    testsets = sorted(os.listdir(f'./inference/test_sets/{folder}/{device_id}/0/l/'))
    print(testsets)
    testset_name = str(input("choose testset : "))

    left_images = sorted(glob.glob(f'./inference/test_sets/{folder}/{device_id}/*/l/{testset_name}/*.jpg'))
    right_images = sorted(glob.glob(f'./inference/test_sets/{folder}/{device_id}/*/r/{testset_name}/*.jpg'))
    
    for left_img, right_img in zip(left_images, right_images):
        inferencer = inferences
        floor = left_img.split('/')[-4]

        left_em_boxes = get_boxes(f'./inference/boxes/{folder}/{device_id}/{floor}/l/empty.xml')
        left_main_boxes = get_boxes(f'./inference/boxes/{folder}/{device_id}/{floor}/l/main.xml')

        right_em_boxes = get_boxes(f'./inference/boxes/{folder}/{device_id}/{floor}/r/empty.xml')
        right_main_boxes = get_boxes(f'./inference/boxes/{folder}/{device_id}/{floor}/r/main.xml')

        left_image = cv2.imread(left_img)
        right_image = cv2.imread(right_img)
        left_main_images, left_em_images = img_preprocess(left_main_boxes, left_em_boxes, left_image)
        right_main_images, right_em_images = img_preprocess(right_main_boxes, right_em_boxes, right_image)

        main_merge = left_main_images + right_main_images
        empty_merge = left_em_images + right_em_images
        spliter.append(len(main_merge))

        empty_final = empty_final + empty_merge
        main_final = main_final + main_merge

    empty_final = np.array(empty_final)
    main_final = np.array(main_final)
    print(main_final.shape, empty_final.shape)

    empty_pred = inferencer.predict(empty_final, 'empty')
    main_pred = inferencer.predict(main_final, 'main')

    inf_result = list(map(lambda x: x[0] if x[0] == 'empty' else x[1] , zip(empty_pred, main_pred)))
    
    for left_img, right_img in zip(left_images, right_images):
        os.system('clear')
        print(left_img, ' / ', right_img)

        floor = left_img.split('/')[-4]
        left_em_boxes = get_boxes(f'./inference/boxes/{folder}/{device_id}/{floor}/l/empty.xml')
        left_main_boxes = get_boxes(f'./inference/boxes/{folder}/{device_id}/{floor}/l/main.xml')
        right_em_boxes = get_boxes(f'./inference/boxes/{folder}/{device_id}/{floor}/r/empty.xml')
        right_main_boxes = get_boxes(f'./inference/boxes/{folder}/{device_id}/{floor}/r/main.xml')

        coulmns = len(left_main_boxes) + len(right_main_boxes)
        tmp_res = inf_result[:coulmns]
        print(tmp_res)
        inf_result = inf_result[coulmns:]

        left_img = cv2.imread(left_img)
        right_img = cv2.imread(right_img)

        left_show = left_img.copy()
        right_show = right_img.copy()

        for (xmin, ymin, xmax, ymax) in left_main_boxes:
            left_show = cv2.rectangle(left_show, (xmin, ymin), (xmax, ymax), (255, 0, 0), 3)
            left_show = cv2.putText(left_show, str(tmp_res[0]) , (xmin, ymin+30), cv2.FONT_HERSHEY_SIMPLEX, .9, (0, 0, 0), 3)
            del tmp_res[0]
        for (xmin, ymin, xmax, ymax) in left_em_boxes:
            left_show = cv2.rectangle(left_show, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
        
        for (xmin, ymin, xmax, ymax) in right_main_boxes:
            right_show = cv2.rectangle(right_show, (xmin, ymin), (xmax, ymax), (255, 0, 0), 3)
            right_show = cv2.putText(right_show, str(tmp_res[0]) , (xmin, ymin+30), cv2.FONT_HERSHEY_SIMPLEX, .9, (0, 0, 0), 3)
            del tmp_res[0]
        for (xmin, ymin, xmax, ymax) in right_em_boxes:
            right_show = cv2.rectangle(right_show, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)

        left_show = cv2.resize(left_show, (940, 700))
        right_show = cv2.resize(right_show, (940, 700))
        concat_frame = np.concatenate((left_show, right_show), axis=1)

        cv2.imshow('Inference Result', concat_frame)
        k = cv2.waitKey(0)
        if k == ord('q'):
            break

        if k == ord('s'):
            save_images(left_main_boxes, right_main_boxes, left_img, right_img)


def getDevicesList():
    os.system('clear')
    devices_list = []

    result = os.popen('v4l2-ctl --list-devices').read()
    result_lists = result.split("\n\n")
    for result_list in result_lists:
        if result_list != '':
            result_list_2 = result_list.split('\n\t')
            devices_list.append(result_list_2[1][-1])

    devices_list = sorted(devices_list)
    print("USB CAM List :", devices_list)
    return devices_list


def inference_video():
    """CAM SETTING"""
    devices_list = getDevicesList()

    right_cam = cv2.VideoCapture(int(devices_list[0]))
    left_cam = cv2.VideoCapture(int(devices_list[1]))

    frame_width, frame_height = int(1920), int(1080)
    MJPG_CODEC = 1196444237.0 # MJPG
    cap_AUTOFOCUS = 0
    cap_FOCUS = 0

    right_cam.set(cv2.CAP_PROP_BRIGHTNESS, 10)
    right_cam.set(cv2.CAP_PROP_FOURCC, MJPG_CODEC)
    right_cam.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    right_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    right_cam.set(cv2.CAP_PROP_AUTOFOCUS, cap_AUTOFOCUS)

    left_cam.set(cv2.CAP_PROP_BRIGHTNESS, 10)
    left_cam.set(cv2.CAP_PROP_FOURCC, MJPG_CODEC)
    left_cam.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    left_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    left_cam.set(cv2.CAP_PROP_AUTOFOCUS, cap_AUTOFOCUS)
    """CAM SETTING"""

    folder = store_name.split('_')[0]
    device_id = os.listdir(f'./inference/test_sets/{folder}/')

    if len(device_id) > 1:
        print(device_id)
        device_id = str(input("choose device_id : "))

    else:
        device_id = device_id[0]

    floor_list = sorted(os.listdir(f'./inference/boxes/{folder}/{device_id}'))
    print('Total Floors :', floor_list)
    floor = int(floor_list[0])
    print("Default Floor is '0'")

    while True:
        empty_final = []
        main_final = []

        _, right_frame = right_cam.read()
        _, left_frame = left_cam.read()

        left_em_boxes = get_boxes(f'./inference/boxes/{folder}/{device_id}/{floor}/l/empty.xml')
        left_main_boxes = get_boxes(f'./inference/boxes/{folder}/{device_id}/{floor}/l/main.xml')
        right_em_boxes = get_boxes(f'./inference/boxes/{folder}/{device_id}/{floor}/r/empty.xml')
        right_main_boxes = get_boxes(f'./inference/boxes/{folder}/{device_id}/{floor}/r/main.xml')

        inferencer = inferences
        left_main_images, left_em_images = img_preprocess(left_main_boxes, left_em_boxes, left_frame)
        right_main_images, right_em_images = img_preprocess(right_main_boxes, right_em_boxes, right_frame)

        main_merge = left_main_images + right_main_images
        empty_merge = left_em_images + right_em_images

        empty_final = empty_final + empty_merge
        main_final = main_final + main_merge

        empty_final = np.array(empty_final)
        main_final = np.array(main_final)
        # print(main_final.shape, empty_final.shape)

        empty_pred = inferencer.predict(empty_final, 'empty')
        main_pred = inferencer.predict(main_final, 'main')

        inf_result = list(map(lambda x: x[0] if x[0] == 'empty' else x[1] , zip(empty_pred, main_pred)))
        os.system('clear')
        print('Current floor : ', floor)
        print(inf_result)

        if right_cam:
            right_show = right_frame.copy()
            for (xmin, ymin, xmax, ymax) in right_main_boxes:
                right_show = cv2.rectangle(right_show, (xmin, ymin), (xmax, ymax), (255, 0, 0), 3)
            for (xmin, ymin, xmax, ymax) in right_em_boxes:
                right_show = cv2.rectangle(right_show, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
            right_show = cv2.resize(right_show, (960, 960))

        if left_cam:
            left_show = left_frame.copy()
            for (xmin, ymin, xmax, ymax) in left_main_boxes:
                left_show = cv2.rectangle(left_show, (xmin, ymin), (xmax, ymax), (255, 0, 0), 3)
            for (xmin, ymin, xmax, ymax) in left_em_boxes:
                left_show = cv2.rectangle(left_show, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
            left_show = cv2.resize(left_show, (960, 960))

        concat_frame = np.concatenate((left_show, right_show), axis=1)
        cv2.imshow("LEFT FRAME / RIGHT FRAME", concat_frame)

        k = cv2.waitKey(1)
        if k == ord('q'):
            break

        elif k == ord('d'):
            if floor < len(floor_list) - 1:
                floor += 1

            else:
                floor = 0

        elif k == ord('a'):
            if floor > 0:
                floor -= 1

            else:
                floor = len(floor_list) - 1

        elif k == ord('s'):
            save_images(left_main_boxes, right_main_boxes, left_frame, right_frame)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dual CAM Inference")
    parser.add_argument('--store_name', type=str)
    parser.add_argument('--inference_type', type=str)
    args = parser.parse_args()

    store_name = args.store_name
    model_path = './inference/model_and_label/' + store_name
    inferences =  InferenceClass(model_path + "/empty_model.h5",
                                 model_path + "/empty_labels.txt",
                                 model_path + "/main_model.h5",
                                 model_path + "/main_labels.txt")
                                 
    if args.inference_type == 'images':
        inference_test_images()

    elif args.inference_type == 'video':
        inference_video()

    else:
        print('This type is not supported.')