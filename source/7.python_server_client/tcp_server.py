import socket

HOST = '221.148.121.77' # 접속할 서버 주소. 여기에서는 루프백(loopback) 인터페이스 주소 즉 localhost를 사용.    
PORT = 1025 # 클라이언트 접속을 대기하는 포트 번호.
  
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM) # 소켓 객체를 생성. 주소 체계(address family)로 IPv4, 소켓 타입으로 TCP 사용. 
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) # 포트 사용중이라 연결할 수 없다는 WinError 10048 에러 해결를 위해 필요.


# bind 함수는 소켓을 특정 네트워크 인터페이스와 포트 번호에 연결하는데 사용.
# HOST는 hostname, ip address, 빈 문자열 ""이 될 수 있고 빈 문자열이면 모든 네트워크 인터페이스로부터의 접속을 허용
# PORT는 1-65535 사이의 숫자를 사용할 수 있다.  
server_socket.bind((HOST, PORT)) 
server_socket.listen() # 서버가 클라이언트의 접속을 허용하도록 합니다. 
client_socket, addr = server_socket.accept() # accept 함수에서 대기하다가 클라이언트가 접속하면 새로운 소켓을 리턴.
print('Connected by', addr) # 접속한 클라이언트의 주소입니다.

while True:

    # 클라이언트가 보낸 메시지를 수신하기 위해 대기.
    data = client_socket.recv(1024)
 
    if not data: # 빈 문자열을 수신하면 루프를 중지.
        break

    print('Received from', addr, data.decode()) # 수신받은 문자열을 출력. 
    client_socket.sendall(data) # 받은 문자열을 다시 클라이언트로 전송해줍니다.(에코)

# 소켓을 닫음.
client_socket.close()
server_socket.close()