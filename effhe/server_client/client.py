import socket
import struct
from time import sleep
import torch
import json
import tenseal as ts
from torchvision import datasets
import torchvision.transforms as transforms
from effhe.server_client.data import get_MNIST_test_loader, data_to_list, get_query_data
from effhe.server_client.cryptography import gen_key, encrypt_data, make_public_key
from effhe.constants.server_client import HOST, PORT, MODEL, HEADER_LENGTH
import sys




# --------------------------------------------------------------------------
#------------------------Design Message Protocol----------------------------
#---------------------------------------------------------------------------

class Client():
    def __init__(self):
        self.create_client()


    def create_client(self):
        # Create client socket
        self.client = socket.socket()        
    
        # establish LOCAL connection
        self.client.connect((HOST, PORT))


    def receive_fixed_length_msg(self, msglen):
        message = b''
        while len(message) < msglen:
            chunk = self.client.recv(msglen - len(message))
            if chunk == b'':
                raise RuntimeError("socket connection broken")
            message = message + chunk
        return message

    def receive_message(self):
        header = self.receive_fixed_length_msg(HEADER_LENGTH)
        message_length = struct.unpack("!Q", header)[0] 

        message = None
        if message_length > 0: 
            message = self.receive_fixed_length_msg(message_length) 
            message = message.decode("utf-8")

        return message

    def send_message(self, message, preencoded=False):
        if not preencoded:
            message = message.encode("utf-8") 
        header = struct.pack("!Q", len(message))

        message = header + message 
        self.client.sendall(message)

    def close(self):
        self.client.close()

    def make_payload(self, data, private_key):
    
        #Make data ingestible
        data=data_to_list(data)

        # Encrypt data
        data_enc, windows_nb = encrypt_data(data=data, context=private_key)

        # Make public key
        public_key  = make_public_key(private_key)

        # Serialise data for payload
        public_key = public_key.serialize()
        data_enc = data_enc.serialize()

        # Create json
        payload = {
            "data": str(data_enc),
            "public_key": str(public_key),
            "model": MODEL
        }

        payload = json.dumps(payload)
        payload = payload.encode('ascii')

        return payload





# --------------------------------------------------------------------------
#------------------------Driver code starts here----------------------------
#---------------------------------------------------------------------------
 
# Generate Key 
private_key = gen_key("small")

# Get dataloader and query data
query = get_query_data()



# Create client
c = Client()


# Test connection
response = c.receive_message()
print(response)


# Create payload
payload = c.make_payload(query, private_key)
c.send_message(payload, preencoded=True)

# Wait for affirmation
affirmation = c.receive_message()
if int(affirmation) != 200:
    print(affirmation)
    c.close()

else:
    print("Inference procedure commencing...")

# close the connection
c.close()  

















# Send Payload
# client.send(str(BUFFERSIZE).encode())


# client.send(payload)
# client.send(''.encode())
# print("payload sent.")
# sleep(3)
# client.send('hello'.encode())
# batch = client.recv(1024)
# affirmation = ''
# # Recieve affirmation
# while batch:
#     affirmation += batch.decode()
#     print("getting another")
#     batch = client.recv(1024)
    # if not batch:
    #     break

# print(affirmation)
# if affirmation == "request accepted":
#     print("Request for inference accepted")

# else:
#     print(affirmation)
  