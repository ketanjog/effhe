import torch
from effhe.server_client.data import get_query_data
from effhe.server_client.cryptography import gen_key
from effhe.constants.server_client import TRACK_TIME
from effhe.server_client.message_protocols import Client
from timeit import default_timer
from effhe.server_client.cryptography import make_public_key


# Generate Key
private_key = gen_key("small")

num_samples = 3

for idx in range(num_samples):
    # Get dataloader and query data
    query, label = get_query_data(idx)

    # Create client
    c = Client()
    
    print("=================================")
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
        public_key = make_public_key(private_key)
        print("Inference procedure commencing...")
        relu_time = 0

        start_time = default_timer()

        relu_time += c.do_non_linear(public_key, private_key, track_time = True) #first relu

        relu_time += c.do_non_linear(public_key, private_key, track_time = True) #second relu


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
        if(TRACK_TIME):
            server_time = c.receive_message()
            print("times:", server_time)

        print("prediction:", dec_pred)
        print("ground truth: ", label)
        print("=================================")

        # close the connection
        c.close()