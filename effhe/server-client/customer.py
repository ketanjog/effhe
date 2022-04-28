import socket
import torch
import json
import tenseal as ts

#This is the CUSTOMER file

#They act as the server since they recieve a request, apply non-linearity, and send it back

PORT = 8080
IP = '0.0.0.0'

def compute_act(act, inpt, params=None):
	if(act == "relu"):
		slope = 0

		if(params != None):
			slope = params['slope']

		relu = torch.nn.LeakyReLU(slope)

		return relu(inpt)

	else:
		print("Operation not supported.")
		return None

def get_data(server, byte_size):
	data = server.recv(byte_size)
	
	while(data != None):
		data += server.recv(byte_size)
		#https://stackoverflow.com/questions/24423162/how-to-send-an-array-over-a-socket-in-python
	
	data = json.loads(data.decode())
	params = data['params']
	tensor = data['tensor']
	tensor = torch.tensor(tensor)

	return params, tensor

def gen_key(kind):
	bits_scale = 26

	context = None
	
	if kind == "small":
		context = ts.context(
		    ts.SCHEME_TYPE.CKKS,
		    poly_modulus_degree=8192,
		    coeff_mod_bit_sizes=[31, bits_scale, bits_scale, bits_scale, 31]
		)
	elif kind == "mid" :
		context = ts.context(
		    ts.SCHEME_TYPE.CKKS,
		    poly_modulus_degree=8192,
		    coeff_mod_bit_sizes=[31, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, 31]
		)
	else:
		context = ts.context(
		    ts.SCHEME_TYPE.CKKS,
		    poly_modulus_degree=16384,
		    coeff_mod_bit_sizes=[31, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, 31]
		)

	context.global_scale = pow(2, bits_scale)

	context.generate_galois_keys()

	return context


def get_server():
	server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	server.bind((IP, PORT))
	return server

def get_client():
	client = client.socket(socket.AF_INET, socket.SOCK_STREAM)
	return client

def send_data(client, data, context, kernel_shape, stride):
	client.connect((IP, PORT))

	data_enc, windows_nb = ts.im2col_encoding(
            context, data.view(28, 28).tolist(), kernel_shape[0],
            kernel_shape[1], stride
    )

	data_enc = data_enc.serialize()

	payload = {
		"data": data_enc
	}

	payload = json.dumps(payload)

	client.send(payload)
	client.close()

#Part 1

kernel_shape = (7, 7)
stride = 3

client = get_client()
context = gen_key("small")
send_data(client, None, context, kernel_shape, stride)












