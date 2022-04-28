import torch
import tenseal as ts
import socket
import json

#This is the SERVICE file

#They act as the client since they are requesting non-linearities to be calculated 

PORT = 8080

# Create socket
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

HOST = "127.0.0.1"  # Standard loopback interface address (localhost)
PORT = 65432  # Port to listen on (non-privileged ports are > 1023)

# Listen for client requests:
server.bind((HOST, PORT))
server.listen()
conn, addr = server.accept()
with conn:
    print(f"Connection established by {addr}")

    print("Stage 1: Getting Request Type")
    # Initialise data object
    data = ""
    while True:
        # Read in client request
        req = conn.recv(1024)
        if(len(req) == 0):
        	break
        data += req.decode('ascii')

client_request = json.load(data)

print(client_request)






	