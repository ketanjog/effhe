import tenseal as ts
import torch

class EncConvReluNet:
    def __init__(self, torch_nn, context):
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
        
        
    def forward(self, enc_x, windows_nb, server = None):
        # conv layer
        enc_channels = []
        for kernel, bias in zip(self.conv1_weight, self.conv1_bias):
            y = enc_x.conv2d_im2col(kernel, windows_nb) + bias
            enc_channels.append(y)
        # pack all channels into a single flattened vector
        enc_x = ts.CKKSVector.pack_vectors(enc_channels)

        # =====BEGIN CLIENT-SIDE ACTIONS=====
        # decrypt
        dec_x = enc_x.decrypt()
        dec_x = torch.tensor(dec_x)

        # apply relu
        dec_x = self.relu(dec_x)

        # encrypt again
        enc_x = ts.CKKSVector(self.context, dec_x)

        #=====END CLIENT-SIDE ACTIONS=====

        # fc1 layer
        enc_x = enc_x.mm(self.fc1_weight) + self.fc1_bias
        
        # =====BEGIN CLIENT-SIDE ACTIONS=====
        # decrypt 
        dec_x = enc_x.decrypt()
        dec_x = torch.tensor(dec_x)

        # apply relu
        dec_x = self.relu(dec_x)

        # encrypt again
        enc_x = ts.CKKSVector(self.context, dec_x)

        #=====END CLIENT-SIDE ACTIONS=====
        
        # fc2 layer
        enc_x = enc_x.mm(self.fc2_weight) + self.fc2_bias
        return enc_x
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)