import os

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST

import matplotlib.pyplot as plt

import pytorch_lightning as pl
from lightning.pytorch import LightningModule


'''
Here is how the math works for transpose convolution:
output = (input - 1)stride + kernal

Here is how the math works for convolution:
output = floor((input + 2(padding) - kernal)/ stride) + 1
'''


# Detective: fake or no fake -> 1 output [0, 1]
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        # Simple CNN
        self.norm1 = nn.BatchNorm2d(3)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1) # [64, 32, 32]
        self.conv1_drop = nn.Dropout2d()
        self.norm1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1) # [128, 16, 16]
        self.conv2_drop = nn.Dropout2d()
        self.norm2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1) # [256, 8, 8]
        self.conv3_drop = nn.Dropout2d()
        self.norm3 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(16384, 50)
        self.fc2 = nn.Linear(50, 1)
  
    def forward(self, x):

        x = F.relu(self.conv1_drop(self.conv1(x)))
        x = self.norm1(x)

        x = F.relu(self.conv2_drop(self.conv2(x)))
        x = self.norm2(x)

        x = F.relu(self.conv3_drop(self.conv3(x)))
        x = self.norm3(x)

        # Flatten the tensor so it can be fed into the FC layers
        x = x.view(-1, 16384)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return torch.sigmoid(x)