import cv2
import socket
import numpy as np

def recvall(sock, count):
    buf = b''

    while count:
        newbuf = sock.recv(count)
        if not newbuf:
            return None

        buf += newbuf
        count -= len(newbuf)

    return buf

HOST = 'localhost'
PORT = 8888
  
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(True)

connect, address = server_socket.accept()

length = recvall(connect, 16)
stringData = recvall(connect, int(length))
data = np.frombuffer(stringData, dtype='uint8')

data = cv2.imdecode(data, 1)
cv2.imshow('test', data)
cv2.waitKey(0)

server_socket.close()