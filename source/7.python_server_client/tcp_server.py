import cv2
import pickle
import socket
import struct
import darknet
from ctypes import *

weight_file = "/data/Models/etri_yolov4/yolov4_final.weights"
config_file = "/home/barcelona/darknet/custom/etri/deploy/yolov4.cfg"
data_file = "/home/barcelona/darknet/custom/etri/data/etri.data"
thresh_hold = .6

network, class_names, class_colors = darknet.load_network(config_file, data_file, weight_file, batch_size=1)
width = darknet.network_width(network)
height = darknet.network_height(network)
darknet_image = darknet.make_image(width, height, 3)

HOST = 'localhost'
PORT = 8089

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print('소켓 생성')

s.bind((HOST, PORT))
s.listen(10)

conn, addr = s.accept()

data = b''
payload_size = struct.calcsize("L")

while True:
    while len(data) < payload_size:
        data += conn.recv(4096)

    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack("L", packed_msg_size)[0]

    while len(data) < msg_size:
        data += conn.recv(4096)

    frame_data = data[:msg_size]
    data = data[msg_size:]

    frame = pickle.loads(frame_data)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (width, height))

    darknet.copy_image_from_bytes(darknet_image, frame.tobytes())
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh_hold)
    result = darknet.draw_boxes(detections, frame, class_colors)

    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    cv2.imshow('Detection Result', result)
    cv2.waitKey(1)