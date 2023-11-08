import socket
host = socket.gethostname()
print(host)
host = socket.gethostbyname(host)
print(host)
