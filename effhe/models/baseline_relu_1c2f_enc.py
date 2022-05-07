import tenseal as ts
import torch
from timeit import default_timer

class EncConvReluNet:
    def __init__(self, torch_nn, context=None, use_socket=False, pub_key = None):
        self.conv1_weight = torch_nn.conv1.weight.data.view(
            torch_nn.conv1.out_channels, torch_nn.conv1.kernel_size[0],
            torch_nn.conv1.kernel_size[1]
        ).tolist()
        self.conv1_bias = torch_nn.conv1.bias.data.tolist()
        
        self.fc1_weight = torch_nn.fc1.weight.T.data.tolist()
        self.fc1_bias = torch_nn.fc1.bias.data.tolist()
        
        self.fc2_weight = torch_nn.fc2.weight.T.data.tolist()
        self.fc2_bias = torch_nn.fc2.bias.data.tolist()

        self.context = context
        self.relu = torch.nn.ReLU()
        self.use_socket = use_socket
        self.pub_key = pub_key
        self.time_store = None
        
        
    def forward(self, enc_x, windows_nb, server = None, track_time = False, time_store = None):
        time_val = 0
        #print("begin forward")
        start_time = default_timer()
        # conv layer
        enc_channels = []
        for kernel, bias in zip(self.conv1_weight, self.conv1_bias):
            y = enc_x.conv2d_im2col(kernel, windows_nb) + bias
            enc_channels.append(y)
        # pack all channels into a single flattened vector
        enc_x = ts.CKKSVector.pack_vectors(enc_channels)

        time_val += default_timer() - start_time

        # Conv layer time
        conv_time = default_timer() - start_time

        if(self.use_socket):
            enc_x_bytes = enc_x.serialize()
            server.send_message(enc_x_bytes, preencoded=True)
            enc_x = server.receive_message(decode_bytes=False)
            start_time2 = default_timer()
            enc_x = server.prepare_input(self.pub_key, enc_x)
        else:
            # =====BEGIN CLIENT-SIDE ACTIONS=====
            # decrypt 
            dec_x = torch.tensor(enc_x.decrypt())

            # apply relu
            dec_x = self.relu(dec_x)

            # encrypt again
            enc_x = ts.CKKSVector(self.context, dec_x)

            #=====END CLIENT-SIDE ACTIONS=====

        # fc1 layer
        fc1_begins = default_timer()
        enc_x = enc_x.mm(self.fc1_weight) + self.fc1_bias

        # FC1 time:
        fc1_time = default_timer() - fc1_begins

        time_val += default_timer() - start_time2
        
        if(self.use_socket):
            enc_x_bytes = enc_x.serialize()
            server.send_message(enc_x_bytes, preencoded=True)
            enc_x = server.receive_message(decode_bytes=False)
            start_time2 = default_timer()
            enc_x = server.prepare_input(self.pub_key, enc_x)
        else:
            # =====BEGIN CLIENT-SIDE ACTIONS=====
            # decrypt 
            dec_x = torch.tensor(enc_x.decrypt())

            # apply relu
            dec_x = self.relu(dec_x)

            # encrypt again
            enc_x = ts.CKKSVector(self.context, dec_x)

            #=====END CLIENT-SIDE ACTIONS=====
        
        # fc2 layer
        fc2_begins = default_timer()
        enc_x = enc_x.mm(self.fc2_weight) + self.fc2_bias

        # fc2 time
        fc2_time = default_timer() - fc2_begins
        time_val += default_timer() - start_time2

        tot_time = default_timer() - start_time

        if(track_time):
            print("time taken:", time_val)
            # time_store[0] = time_val
            time_store[0] = [conv_time, fc1_time, fc2_time, tot_time]
        
       

        # Store the time taken:
        self.time_store = [conv_time, fc1_time, fc2_time, tot_time]
        #print("end forward")

        return enc_x
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)