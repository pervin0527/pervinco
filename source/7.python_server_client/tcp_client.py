import cv2
import socket
import numpy as np

HOST = 'localhost'
PORT = 8888
  
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((HOST, PORT))

image_path = "/data/Datasets/testset/ETRI_cropped_large/test_sample_04.jpg"
image = cv2.imread(image_path)
image = cv2.resize(image, (416, 416))
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
result, imgencode = cv2.imencode('.jpg', image, encode_param)
data = np.array(imgencode)
stringData = data.tobytes()

client_socket.send(str(len(stringData)).ljust(16).encode())
client_socket.send(stringData)
client_socket.close()