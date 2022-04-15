import torch
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np

class SimpleConvNet(torch.nn.Module):
    def __init__(self, hidden=64, output=10):
        super(SimpleConvNet, self).__init__()        
        self.conv1 = torch.nn.Conv2d(1, 4, kernel_size=7, padding=0, stride=3)
        self.fc1 = torch.nn.Linear(256, hidden)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        # the model uses the square activation function
        x = self.relu(x)
        # flattening while keeping the batch axis
        x = x.view(-1, 256)
        x = self.fc1(x)
        return x