from curses import window
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

from timeit import default_timer



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

    def receive_message(self, decode_bytes = True):
        header = self.receive_fixed_length_msg(HEADER_LENGTH)
        message_length = struct.unpack("!Q", header)[0] 

        message = None
        if message_length > 0: 
            message = self.receive_fixed_length_msg(message_length) 
            if(decode_bytes):
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
            "model": MODEL,
            "windows_nb" : str(windows_nb)
        }

        payload = json.dumps(payload)
        payload = payload.encode('utf-8')

        return payload, public_key, data_enc

    def prepare_input(self, context: bytes, ckks_vector: bytes) -> ts.CKKSVector:
        # context = context.encode('utf-8')
        # ckks_vector = ckks_vector.encode('utf-8')
        try:
            ctx = ts.context_from(context)
            enc_x = ts.ckks_vector_from(ctx, ckks_vector)
        except:
            raise DeserializationError("cannot deserialize context or ckks_vector")
        try:
            _ = ctx.galois_keys()
        except:
            raise InvalidContext("the context doesn't hold galois keys")

        return enc_x

def do_non_linear(c, public_key, private_key, act="relu", track_time = False):
    enc_x = c.receive_message(decode_bytes=False)

    start_time = default_timer()
    enc_x = c.prepare_input(public_key, enc_x)

    print("decrypting...")
    secret_key = private_key.secret_key()
    dec_x = enc_x.decrypt(secret_key)
    dec_x = torch.tensor(dec_x)

    print("performing non-linear operation...")
    dec_x = torch.nn.ReLU()(dec_x)
    enc_x = ts.CKKSVector(private_key, dec_x)

    if(track_time):
        tot_time = default_timer() - start_time
        print("time spent doing relu:", tot_time)

    print("sending back to client...")
    enc_x = enc_x.serialize()
    c.send_message(enc_x, preencoded=True)


# --------------------------------------------------------------------------
#------------------------Driver code starts here----------------------------
#---------------------------------------------------------------------------
 
# Generate Key 
private_key = gen_key("small")

# Get dataloader and query data
query, label = get_query_data()



# Create client
c = Client()


# Test connection
response = c.receive_message()
print(response)


# Create payload
payload, public_key, data_enc = c.make_payload(query, private_key)

# Send payload, pub_key, data
c.send_message(payload, preencoded=True)
c.send_message(public_key, preencoded=True)
c.send_message(data_enc, preencoded=True)

# Wait for affirmation
affirmation = c.receive_message()
if int(affirmation) != 200:
    print(affirmation)
    c.close()

else:
    print("Inference procedure commencing...")

    start_time = default_timer()

    #Recieve first encrypted data
    # enc_x = c.receive_message(decode_bytes=False)
    # enc_x = c.prepare_input(public_key, enc_x)
    # print("decrypting...")
    # secret_key = private_key.secret_key()
    # dec_x = enc_x.decrypt(secret_key)
    # dec_x = torch.tensor(dec_x)
    # dec_x = torch.nn.ReLU()(dec_x)
    # enc_x = ts.CKKSVector(private_key, dec_x)
    # enc_x = enc_x.serialize()
    # c.send_message(enc_x, preencoded=True)

    do_non_linear(c, public_key, private_key, track_time = True) #first relu

    # enc_x = c.receive_message(decode_bytes=False)
    # enc_x = c.prepare_input(public_key, enc_x)
    # print("decrypting...")
    # secret_key = private_key.secret_key()
    # dec_x = enc_x.decrypt(secret_key)
    # dec_x = torch.tensor(dec_x)
    # dec_x = torch.nn.ReLU()(dec_x)
    # enc_x = ts.CKKSVector(private_key, dec_x)
    # enc_x = enc_x.serialize()
    # c.send_message(enc_x, preencoded=True)

    do_non_linear(c, public_key, private_key, track_time = True) #second relu 

    #Receive and make prediction
    enc_pred = c.receive_message(decode_bytes=False)
    enc_pred = c.prepare_input(public_key, enc_pred)
    secret_key = private_key.secret_key()
    dec_pred = enc_pred.decrypt(secret_key)

    tot_time = default_timer() - start_time

    dec_pred = torch.tensor(dec_pred).view(1, -1)
    _, dec_pred = torch.max(dec_pred, 1)
    dec_pred = dec_pred.item()

    print("time taken:", tot_time)
    print("prediction:", dec_pred)
    print("ground truth: ", label)

# Now the back and forth begins:

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
  