import tenseal as ts
from effhe.models.baseline_relu_1c2f_approx_enc import EncConvReluAprxNet
import torch
from torchvision import datasets
import torchvision.transforms as transforms
from effhe.constants.paths import BASELINE_PATH
from effhe.models.baseline_relu_1c2f import ConvReluNet
from timeit import default_timer
from statistics import mean, pstdev
from tqdm import tqdm

# Constants for the test
NUM_SAMPLES = 2
conv_time = []
relu_1_time = []
fc1_time = []
relu_2_time = []
fc2_time = []
server_round_time = []

# Load the data
test_data = datasets.MNIST('data', train=False, download=True, transform=transforms.ToTensor())
# Load one element at a time
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

model = ConvReluNet()
model.load_state_dict(torch.load(BASELINE_PATH))
criterion = torch.nn.CrossEntropyLoss()

def enc_test(context, model, test_loader, criterion, kernel_shape, stride):
    # initialize lists to monitor test loss and accuracy
    
    
    for sample in tqdm(range(NUM_SAMPLES)):
        data, target =  next(iter(test_loader))

        # Encoding and encryption
        start_time = default_timer()
        x_enc, windows_nb = ts.im2col_encoding(
            context, data.view(28, 28).tolist(), kernel_shape[0],
            kernel_shape[1], stride
        )
        # Encrypted evaluation
        enc_output = model(x_enc, windows_nb)
        _conv_time, _relu_1_time, _fc1_time, _relu_2_time, _fc2_time, _tot_time= model.time_store
        
        # store all timestamps
        conv_time.append(_conv_time)
        relu_1_time.append(_relu_1_time)
        fc1_time.append(_fc1_time)
        relu_2_time.append(_relu_2_time)
        fc2_time.append(_fc2_time)
        server_round_time.append(_tot_time)

    print("Conv_time:         " + "Mean: " + str(round(mean(conv_time),5))+ " StdDev: " + str(round(pstdev(conv_time),5)))
    print("Relu 1 time:       " + "Mean: " + str(round(mean(relu_1_time),5)) +  " StdDev: " + str(round(pstdev(relu_1_time),5)))
    print("fc1 time:          " + "Mean: " + str(round(mean(fc1_time),5)) + " StdDev: " + str(round(pstdev(fc1_time),5)))
    print("relu 2 time:       " + "Mean: " + str(round(mean(relu_2_time),5)) + " StdDev: " + str(round(pstdev(relu_2_time),5)))
    print("fc2 time:          " + "Mean: " + str(round(mean(fc2_time),5)) + " StdDev: " + str(round(pstdev(fc2_time),5)))
    print("server_round_time: " + "Mean: " + str(round(mean(server_round_time),5)) + " StdDev: " + str(round(pstdev(server_round_time),5)))




# required for encoding
kernel_shape = model.conv1.kernel_size
stride = model.conv1.stride[0]

## Encryption Parameters

# controls precision of the fractional part
bits_scale = 26

# Create TenSEAL context
context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=16384, # understand why....
    coeff_mod_bit_sizes=[31, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, 31]
)

# set the scale
context.global_scale = pow(2, bits_scale)

# galois keys are required to do ciphertext rotations
context.generate_galois_keys()

enc_model = EncConvReluAprxNet(model)
enc_test(context, enc_model, test_loader, criterion, kernel_shape, stride)