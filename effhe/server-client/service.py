import torch
import tenseal as ts
import socket
import json

#This is the SERVICE file

#They act as the client since they are requesting non-linearities to be calculated 

PORT = 8080

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

HOST = "127.0.0.1"  # Standard loopback interface address (localhost)
PORT = 65432  # Port to listen on (non-privileged ports are > 1023)

server.bind((HOST, PORT))
server.listen()
conn, addr = server.accept()
with conn:
    print(f"Connection established by {addr}")
    while True:
        data += conn.recv(1024)
        if not data:
            break




	