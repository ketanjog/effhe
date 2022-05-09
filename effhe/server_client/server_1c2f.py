from effhe.constants.server_client import HOST, PORT, AVAILABLE_MODELS, HEADER_LENGTH
from effhe.constants.paths import SAVE_PATH, BASELINE_PATH
import json
import torch
from effhe.models.baseline_relu_1c2f import ConvReluNet
from effhe.constants.paths import BASELINE_PATH
from effhe.models.baseline_relu_1c2f_enc import EncConvReluNet
from effhe.server_client.data import train
from effhe.constants.server_client import TRACK_TIME
from effhe.server_client.message_protocols import Server
import tenseal as ts

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

    public_key = ts.context_from(public_key)
    print("Data received!")

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
    pred = enc_model(enc_x, windows_nb, server = s, track_time = True, time_store = time_val)

    pred_bytes = pred.serialize()

    s.send_message(pred_bytes, preencoded=True)
    if(TRACK_TIME):
        time_str = str(time_val[0])
        s.send_message(time_str, preencoded=False)

    print(str(enc_model.time_store))

    print("prediction made!")

