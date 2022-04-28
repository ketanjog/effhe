import torch
import tenseal as ts
import socket
import json

#This is the SERVICE file

#They act as the client since they are requesting non-linearities to be calculated 

PORT = 8080

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

client = client.connect(('0.0.0.0', PORT))
data = client.recv(1024)



	