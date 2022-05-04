import socket
import struct
from effhe.constants.server_client import HOST, PORT, AVAILABLE_MODELS, HEADER_LENGTH
from effhe.constants.paths import SAVE_PATH, BASELINE_PATH
import json
import os
from collections import defaultdict
import sys
import pickle

import tenseal as ts
import torchvision.transforms as transforms
import torch

from effhe.models.baseline_relu_1c2f import ConvReluNet
from effhe.constants.paths import BASELINE_PATH
from effhe.models.baseline_relu_1c2f_enc import EncConvReluNet
from effhe.server_client.data import train
from effhe.constants.server_client import TRACK_TIME
from time import sleep

# --------------------------------------------------------------------------
#------------------------Design Message Protocol----------------------------
#---------------------------------------------------------------------------

# Constants for the protocol:


class Server():
    def __init__(self):

        self.init_data()
        self._create_server()

    def init_data(self):
        """
        Initialise available model status
        """
        # Establish status of each model
        self.trained_models = defaultdict(lambda: "untrained")
        for model in AVAILABLE_MODELS:
            for filename in os.listdir(SAVE_PATH):
                if filename.split('_')[0].lower() == model.lower():
                    self.trained_models[model] = os.path.join(SAVE_PATH, filename)

    def _create_server(self):
        self.server = socket.socket()
        print("Socket creation successful!")

        # Bind to null host - listen for all incoming connections
        self.server.bind((HOST, PORT))

        #Listen for clients. Keep upto 2 clients in the buffer (not really required)
        self.server.listen(2)

    def receive_fixed_length_msg(self, msglen):
        message = b''
        while len(message) < msglen:
            chunk = self.conn.recv(msglen - len(message))
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
        self.conn.sendall(message)

    def accept(self):
        self.conn, self.address = self.server.accept()

    def close(self):
        self.conn.close()

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
# --------------------------------------------------------------------------
#------------------------Driver code starts here----------------------------
#---------------------------------------------------------------------------

s = Server()

while True:

    # Establish connection with a client
    s.accept()
    print("Connected to client at {}".format(s.address))

    # Send test message
    s.send_message('You are connected to EFFHE')

    print("Receiving data...")

    payload = s.receive_message()

    #TenSEAL APIs take in raw bytes so no need to convert to string
    public_key = s.receive_message(decode_bytes=False)
    data_enc = s.receive_message(decode_bytes=False)

    print("Data received!")

    # Recieve payload
    # payload = ''
    # BUFFER = int(client.recv(BUFFER_SIZE).decode())
    # while BUFFER != 0:
    #     batch = client.recv(BUFFER_SIZE).decode()
    #     payload += batch
    #     BUFFER -= sys.getsizeof(batch)
    #     print(BUFFER)

    payload = json.loads(payload)
    
    model = payload["model"]
    print(model)

    # Check whether model is available
    print("Commencing checks...")
    if model not in AVAILABLE_MODELS:
        print("Model not supported. Aborting...")
        
        s.send_message('Requested model not available')
        s.close()
        break

    # If model is untrained, train it
    if s.trained_models[model] == "untrained":
        print("Training model for inference on {}".format(model))
        train(model)

    py_model = None
    if(model == "MNIST"):
        py_model = ConvReluNet()
        py_model.load_state_dict(torch.load(BASELINE_PATH))

    enc_model = EncConvReluNet(py_model, use_socket=True, pub_key = public_key)

    print("Model loaded.")
    
    # Everything is in order to begin inference
    s.send_message('200')

    # Now the back and forth begins
    
    enc_x = s.prepare_input(public_key, data_enc)
    windows_nb = int(payload["windows_nb"])

    time_val = [0]
    pred = enc_model(enc_x, windows_nb, s, track_time = True, time_store = time_val)

    pred_bytes = pred.serialize()

    s.send_message(pred_bytes, preencoded=True)
    if(TRACK_TIME):
        time_str = str(time_val[0])
        s.send_message(time_str, preencoded=False)

    print("prediction made!")

    



    # Close the connection
    #s.close()
    #break



