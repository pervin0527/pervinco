import cv2
import socket
import numpy as np
import darknet
from ctypes import *

def recvall(sock, count):
    buf = b''

    while count:
        newbuf = sock.recv(count)
        if not newbuf:
            return None

        buf += newbuf
        count -= len(newbuf)

    return buf

weight_file = "/data/Models/etri_yolov4/yolov4_final.weights"
config_file = "/home/barcelona/darknet/custom/etri/deploy/yolov4.cfg"
data_file = "/home/barcelona/darknet/custom/etri/data/etri.data"
thresh_hold = .4

network, class_names, class_colors = darknet.load_network(config_file, data_file, weight_file, batch_size=1)

width = darknet.network_width(network)
height = darknet.network_height(network)
darknet_image = darknet.make_image(width, height, 3)

HOST = 'localhost'
PORT = 7777

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(True)

while True:  
    connect, address = server_socket.accept()
    
    length = recvall(connect, 16)
    stringData = recvall(connect, int(length))
    data = np.frombuffer(stringData, dtype='uint8')
    image = cv2.imdecode(data, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (width, height))

    darknet.copy_image_from_bytes(darknet_image, image.tobytes())
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh_hold)
    image = darknet.draw_boxes(detections, image, class_colors)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    cv2.imshow('test', image)
    cv2.waitKey(1)

# server_socket.close()