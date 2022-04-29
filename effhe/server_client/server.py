import socket
from effhe.constants.server_client import HOST, PORT
import json

server = socket.socket()
print("Socket creation successful!")

# Bind to null host - listen for all incoming connections
server.bind((HOST, PORT))

#Listen for clients. Keep upto 2 clients in the buffer (not really required)
server.listen(2)

while True:

    # Establish connection with a client
    client, address = server.accept()
    print("Connected to client at {}".format(address))

    # Send test message
    client.send('You are connected to EFFHE'.encode())

    # Recieve payload
    payload = ''
    while True:
        batch = client.recv(1024)
        payload += batch.decode()

        if not batch:
            break
    # Close the connection

    payload = json.loads(payload)
    print(payload["model"])
    client.close()

    # break


