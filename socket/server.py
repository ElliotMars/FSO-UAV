import socket

# 服务器端

server = socket.socket()
server.bind(('26.170.196.59', 1234))  # 绑定要监听的端口
print("waiting")
server.listen()  # 监听

conn, addr = server.accept()  # 等连接进来
# conn就是客户端连过来而在服务器端为其生成的一个连接实例
print(conn, addr)

while True:
    # data = server.recv(1024)
    data = conn.recv(1024)
    # server.recv(1024)
    print('recv:', data)
    #RL() 