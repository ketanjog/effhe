import torch
import tenseal as ts
from torchvision import datasets
import torchvision.transforms as transforms

def get_MNIST_test_loader():
    # Load the data
    test_data = datasets.MNIST('data', train=False, download=True, transform=transforms.ToTensor())

    # Load one element at a time
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    return test_loader

def data_to_list(x, default_dims = [28,28]):
    """
    Creates list object from tensor
    """
    d1 = default_dims[0]
    d2 = default_dims[1]

    return x.view(d1, d2).tolist()

def get_query_data():
    test_data = datasets.MNIST('data', train=False, download=True, transform=transforms.ToTensor())

    query_data = test_data.__getitem__(0)[0]
    
    return query_data