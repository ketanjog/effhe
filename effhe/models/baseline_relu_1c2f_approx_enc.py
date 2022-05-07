import tenseal as ts
from timeit import default_timer

class EncConvReluAprxNet:
    def __init__(self, torch_nn):
        self.conv1_weight = torch_nn.conv1.weight.data.view(
            torch_nn.conv1.out_channels, torch_nn.conv1.kernel_size[0],
            torch_nn.conv1.kernel_size[1]
        ).tolist()
        self.conv1_bias = torch_nn.conv1.bias.data.tolist()
        
        self.fc1_weight = torch_nn.fc1.weight.T.data.tolist()
        self.fc1_bias = torch_nn.fc1.bias.data.tolist()
        
        self.fc2_weight = torch_nn.fc2.weight.T.data.tolist()
        self.fc2_bias = torch_nn.fc2.bias.data.tolist()
        self.time_store = None
        
        
    def forward(self, enc_x, windows_nb):
        
        # Start time
        start_time = default_timer()

        # conv layer
        enc_channels = []
        for kernel, bias in zip(self.conv1_weight, self.conv1_bias):
            y = enc_x.conv2d_im2col(kernel, windows_nb) + bias
            enc_channels.append(y)
        # pack all channels into a single flattened vector
        enc_x = ts.CKKSVector.pack_vectors(enc_channels)

        # Conv layer time
        conv_time = default_timer() - start_time

        relu_1_start = default_timer()
        #Relu approximation from https://arxiv.org/pdf/1811.09953.pdf (faster cryptonets)
        enc_x = enc_x.polyval(
            [0.25, 
            0.5, 
            0.125]
        ) 
        relu_1_time = default_timer() - relu_1_start

        # fc1 layer
        fc1_begins = default_timer()
        enc_x = enc_x.mm(self.fc1_weight) + self.fc1_bias
        # fc1 time:
        fc1_time = default_timer() - fc1_begins


        relu_2_start = default_timer()
        #Relu approximation from https://arxiv.org/pdf/1811.09953.pdf (faster cryptonets)
        enc_x = enc_x.polyval(
            [0.25, 
            0.5, 
            0.125]
        )
        relu_2_time = default_timer() - relu_2_start

        # fc2 layer
        fc2_begins = default_timer()
        enc_x = enc_x.mm(self.fc2_weight) + self.fc2_bias
        # fc2 time:
        fc2_time = default_timer() - fc2_begins
        tot_time = default_timer() - start_time
        self.time_store = [conv_time, relu_1_time, fc1_time, relu_2_time, fc2_time, tot_time]
        return enc_x
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)