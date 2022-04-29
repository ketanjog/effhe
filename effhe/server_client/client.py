import socket
from webbrowser import get
import torch
import json
import tenseal as ts
from torchvision import datasets
import torchvision.transforms as transforms
from effhe.server_client.data import get_MNIST_test_loader, data_to_list, get_query_data
from effhe.server_client.cryptography import gen_key, encrypt_data, make_public_key
from effhe.constants.server_client import HOST, PORT, MODEL


def make_payload(data, private_key):
    
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


def create_client():
    # Create client socket
    client = socket.socket()        
  
    # establish LOCAL connection
    client.connect((HOST, PORT))

    return client






# --------------------------------------------------------------------------
#------------------------Driver code starts here----------------------------
#---------------------------------------------------------------------------
 
# Generate Key 
private_key = gen_key("small")

# Get dataloader and query data
query = get_query_data()

# Create payload
payload = make_payload(query, private_key)

# Create client
client = create_client()

# Test connection
response = client.recv(1024).decode()
print(response)

# Send Payload
client.send(payload)

# close the connection
client.close()    