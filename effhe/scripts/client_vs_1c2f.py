"""
A script that measures the time and accuracy improvement 
of the Client-Server networks over fully encrypted approximated networks
"""
from effhe.server_client.message_protocols import Server, Client
import torch
from effhe.server_client.data import get_query_data
from effhe.server_client.cryptography import gen_key
from effhe.server_client.cryptography import make_public_key
from effhe.constants.server_client import TRACK_TIME
from effhe.server_client.message_protocols import Client
from timeit import default_timer
import ast
from statistics import pstdev, mean
from tqdm import tqdm
import tenseal as ts


# Generate Key
private_key = gen_key("small")

# Constants for experiment
NUM_SAMPLES = 100
handshake_time = []
decryption_one_time = []
decryption_two_time = []
relu_one_time = []
relu_two_time = []
encryption_one_time = []
encryption_two_time = []
client_one_round_time = []
client_two_round_time = []
inference_time = []
conv_time = []
fc1_time = []
fc2_time = []
server_round_time = []



for idx in tqdm(range(NUM_SAMPLES)):
    iter_begins = default_timer()
    # Get dataloader and query data
    query, label = get_query_data(idx)

    # Create client
    c = Client()
    # Test connection
    response = c.receive_message()
    # print(response)
    # Create payload
    payload, public_key, data_enc = c.make_payload(query, private_key)
    # Send payload, pub_key, data
    c.send_message(payload, preencoded=True)
    c.send_message(public_key, preencoded=True)
    c.send_message(data_enc, preencoded=True)

    # Wait for affirmation
    affirmation = c.receive_message()
    inf_start = default_timer()
    if int(affirmation) != 200:
        print(affirmation)
        c.close()

    else:
        public_key = make_public_key(private_key)
        #print("Inference procedure commencing...")
        handshake_complete = default_timer()
        handshake_time.append(handshake_complete - iter_begins)

        # Do first non linear and log all times
        _decryption_time, _relu_time, _encryption_time, _tot_time = c.do_non_linear(public_key, private_key, track_time = True, exp=True) #first relu
        decryption_one_time.append(_decryption_time)
        relu_one_time.append(_relu_time)
        encryption_one_time.append(_encryption_time)
        client_one_round_time.append(_tot_time)


        # Do second non linear and log all times
        _decryption_time, _relu_time, _encryption_time, _tot_time= c.do_non_linear(public_key, private_key, track_time = True, exp=True) #second relu
        decryption_two_time.append(_decryption_time)
        relu_two_time.append(_relu_time)
        encryption_two_time.append(_encryption_time)
        client_two_round_time.append(_tot_time)        
        
        #Receive and make prediction
        enc_pred = c.receive_message(decode_bytes=False)
        enc_pred = c.prepare_input(public_key, enc_pred)
        secret_key = private_key.secret_key()
        dec_pred = enc_pred.decrypt(secret_key)

        dec_pred = torch.tensor(dec_pred).view(1, -1)
        _, dec_pred = torch.max(dec_pred, 1)
        dec_pred = dec_pred.item()

        # print("time taken:", tot_time)
        if(TRACK_TIME):
            server_time = c.receive_message()
            # Log the server side times
            _conv, _fc1, _fc2, _serv_total = ast.literal_eval(server_time)
            conv_time.append(_conv)
            fc1_time.append(_fc1)
            fc2_time.append(_fc2)
            server_round_time.append(_serv_total)
        
        #print("prediction:",dec_pred)
        #print("label", label)

        # Log total time taken
        tot_time = default_timer() - inf_start
        inference_time.append(tot_time)

        #print(_serv_total)
        #print(tot_time)
            
        # close the connection
        c.close()

print("handshake_time:        " + "Mean: " + str(round(mean(handshake_time),5))+ " StdDev: " + str(round(pstdev(handshake_time),5)))
print("decryption_one_time:       " + "Mean: " + str(round(mean(decryption_one_time),5)) +  " StdDev: " + str(round(pstdev(decryption_one_time),5)))
print("decryption_two_time:       " + "Mean: " + str(round(mean(decryption_two_time),5)) +  " StdDev: " + str(round(pstdev(decryption_two_time),5)))
print("relu_one_time:             " + "Mean: " + str(round(mean(relu_one_time),5)) + " StdDev: " + str(round(pstdev(relu_one_time),5)))
print("relu_two_time:             " + "Mean: " + str(round(mean(relu_two_time),5)) + " StdDev: " + str(round(pstdev(relu_two_time),5)))
print("encryption_one_time:       " + "Mean: " + str(round(mean(encryption_one_time),5)) + " StdDev: " + str(round(pstdev(encryption_one_time),5)))
print("encryption_two_time:       " + "Mean: " + str(round(mean(encryption_two_time),5)) + " StdDev: " + str(round(pstdev(encryption_two_time),5)))
print("client_one_round_time: " + "Mean: " + str(round(mean(client_one_round_time),5)) + "StdDev: " + str(round(pstdev(client_one_round_time),5)))
print("client_two_round_time: " + "Mean: " + str(round(mean(client_two_round_time),5)) + "StdDev: " + str(round(pstdev(client_two_round_time),5)))
print("inference_time:        " + "Mean: " + str(round(mean(inference_time),5)) + " StdDev: " + str(round(pstdev(inference_time),5)))
print("conv_time:             " + "Mean: " + str(round(mean(conv_time),5)) + " StdDev: " + str(round(pstdev(conv_time),5)))
print("fc1_time:              " + "Mean: " + str(round(mean(fc1_time),5)) + " StdDev: " + str(round(pstdev(fc1_time),5)))
print("fc2_time:              " + "Mean: " + str(round(mean(fc2_time),5)) + " StdDev: " + str(round(pstdev(fc2_time),5)))
print("server_round_time:     " + "Mean: " + str(round(mean(server_round_time),5)) + " StdDev: " + str(round(pstdev(server_round_time),5)))

