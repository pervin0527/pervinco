import cv2
import numpy as np
import socket
import sys
import pickle
import struct

# 비디오 경로 읽어오기
###############################################################################################
cap = cv2.VideoCapture(-1)
MJPG_CODEC = 1196444237.0 # MJPG
cap_AUTOFOCUS = 0
cap_FOCUS = 0
#cap_ZOOM = 400

frame_width = int(640)
frame_height = int(480)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
# cv2.namedWindow('inference', cv2.WINDOW_FREERATIO)
# cv2.resizeWindow('inference', frame_width, frame_height)

cap.set(cv2.CAP_PROP_BRIGHTNESS, 100)
# cap.set(cv2.CAP_PROP_FOURCC, MJPG_CODEC)
cap.set(cv2.CAP_PROP_AUTOFOCUS, cap_AUTOFOCUS)
cap.set(cv2.CAP_PROP_FOCUS, cap_FOCUS)
##############################################################################################
clientsocket=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
clientsocket.connect(('localhost',8089))

while True:
    ret,frame = cap.read()
    data = pickle.dumps(frame)
    message_size = struct.pack("L", len(data))
    clientsocket.sendall(message_size + data)