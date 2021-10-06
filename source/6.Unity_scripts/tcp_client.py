import socket

HOST = '221.148.121.77'  # 서버의 주소. hostname 또는 ip address를 사용할 수 있다.
PORT = 1025 # 서버에서 지정한 포트 번호. 
  
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM) # 소켓 객체를 생성. 주소 체계(address family)로 IPv4, 소켓 타입으로 TCP 사용.
client_socket.connect((HOST, PORT)) # 지정한 HOST와 PORT를 사용하여 서버에 접속. 
client_socket.sendall('안녕'.encode()) # 메시지를 전송

data = client_socket.recv(1024) # 메시지를 수신. 
print('Received', repr(data.decode()))

# 소켓을 닫음.
client_socket.close()