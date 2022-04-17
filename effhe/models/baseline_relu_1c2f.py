import torch
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np

class ConvReluNet(torch.nn.Module):
    def __init__(self, hidden=64, output=10):
        super(ConvReluNet, self).__init__()        
        self.conv1 = torch.nn.Conv2d(1, 4, kernel_size=7, padding=0, stride=3)
        self.fc1 = torch.nn.Linear(256, hidden)
        self.fc2 = torch.nn.Linear(hidden, output)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        # the model uses the square activation function
        x = self.relu(x)
        # flattening while keeping the batch axis
        x = x.view(-1, 256)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x