import cv2
import socket
import numpy as np

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

HOST = 'localhost'
PORT = 7777
  
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((HOST, PORT))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("nbbbbbbb")
        break

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    result, imgencode = cv2.imencode('.jpg', frame, encode_param)
    data = np.array(imgencode)
    stringData = data.tobytes()

    client_socket.send(str(len(stringData)).ljust(16).encode())
    client_socket.send(stringData)
    # client_socket.close()