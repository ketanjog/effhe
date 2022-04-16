import tenseal as ts
import torch

class EncSimpleAprxConvNet:
    '''
    Note that this requires context for init. However, in practice this is not
    needed as the client will be the one decrypting and re-encrypting before sending
    the result back to the model during the forward pass.
    '''

    def __init__(self, torch_nn, context):
        self.conv1_weight = torch_nn.conv1.weight.data.view(
            torch_nn.conv1.out_channels, torch_nn.conv1.kernel_size[0],
            torch_nn.conv1.kernel_size[1]
        ).tolist()
        self.conv1_bias = torch_nn.conv1.bias.data.tolist()
        
        self.fc1_weight = torch_nn.fc1.weight.T.data.tolist()
        self.fc1_bias = torch_nn.fc1.bias.data.tolist()

        self.relu = torch.nn.ReLU()
        self.context = context 
        
    def forward(self, enc_x, windows_nb):
        # conv layer
        enc_channels = []
        for kernel, bias in zip(self.conv1_weight, self.conv1_bias):
            y = enc_x.conv2d_im2col(kernel, windows_nb) + bias
            enc_channels.append(y)

        # pack all channels into a single flattened vector
        enc_x = ts.CKKSVector.pack_vectors(enc_channels)

        #Relu approximation from https://arxiv.org/pdf/1811.09953.pdf (faster cryptonets)
        enc_x = 0.125*enc_x*enc_x + 0.5*enc_x + 0.25

        # fc1 layer
        enc_x = enc_x.mm(self.fc1_weight) + self.fc1_bias

        return enc_x
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)