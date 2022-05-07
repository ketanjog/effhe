import socket
import struct
from effhe.constants.server_client import HOST, PORT, AVAILABLE_MODELS, HEADER_LENGTH, MODEL
from effhe.constants.paths import SAVE_PATH, BASELINE_PATH
from effhe.server_client.cryptography import encrypt_data, make_public_key
from effhe.server_client.data import get_MNIST_test_loader, data_to_list, get_query_data
import json
import torch
from timeit import default_timer

from collections import defaultdict
import os
import tenseal as ts

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
        self.active_clients = defaultdict()

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
            raise ValueError("cannot deserialize context or ckks_vector")
        try:
            _ = ctx.galois_keys()
        except:
            raise ValueError("the context doesn't hold galois keys")

        return enc_x

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
        
        self.public_key = public_key

        return payload, public_key, data_enc

    def prepare_input(self, context: bytes, ckks_vector: bytes) -> ts.CKKSVector:
        # context = context.encode('utf-8')
        # ckks_vector = ckks_vector.encode('utf-8')
        try:
            ctx = ts.context_from(context)
            enc_x = ts.ckks_vector_from(ctx, ckks_vector)
        except:
            raise ValueError("cannot deserialize context or ckks_vector")
        try:
            _ = ctx.galois_keys()
        except:
            raise ValueError("the context doesn't hold galois keys")

        return enc_x

    def do_non_linear(self, public_key, private_key, act="relu", track_time = False, exp=False, verbose=False):
        verboseprint = print if verbose else lambda *a, **k: None

        enc_x = self.receive_message(decode_bytes=False)

        # Start time
        start_time = default_timer()

        enc_x = self.prepare_input(public_key, enc_x)

        verboseprint("decrypting...")
        secret_key = private_key.secret_key()
        dec_x = enc_x.decrypt(secret_key)
        dec_x = torch.tensor(dec_x)

        # Decryption time
        decryption_time = default_timer() - start_time

        verboseprint("performing non-linear operation...")
        dec_x = torch.nn.ReLU()(dec_x)

        # Relu time
        relu_time = default_timer() - decryption_time - start_time

        verboseprint("sending back to client...")
        enc_x = ts.CKKSVector(private_key, dec_x)
        enc_x = enc_x.serialize()

        # Encryption time
        encryption_time = default_timer() - relu_time - decryption_time - start_time
        self.send_message(enc_x, preencoded=True)

        tot_time = default_timer() - start_time
        if(track_time):
            verboseprint("time spent doing relu:", tot_time)

        if(exp):
            return decryption_time, relu_time, encryption_time, tot_time

        if(track_time):
            return tot_time
        else:
            return None