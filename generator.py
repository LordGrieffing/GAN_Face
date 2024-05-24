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

# Generate Fake Data: output like real data [1, 28, 28] and values -1, 1
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.lin1 = nn.Linear(latent_dim, 8192)  # [n, 512, 4, 4]
        self.norm1 = nn.BatchNorm2d(512)
        self.ct1 = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1) # [n, 256, 8, 8]
        self.norm2 = nn.BatchNorm2d(256)
        self.ct2 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1) # [n, 128, 16, 16]
        self.norm3 = nn.BatchNorm2d(128)
        self.ct3 = nn.ConvTranspose2d(128, 96, 4, stride=2, padding=1) # [n, 96, 32, 32]
        self.norm4 = nn.BatchNorm2d(96)
        self.ct4 = nn.ConvTranspose2d(96, 3, 4, stride=2, padding=1) # [n, 3, 64, 64]
    

    def forward(self, x):
        # Pass latent space input into linear layer and reshape
        x = self.lin1(x)
        x = F.gelu(x)
        x = x.view(-1, 512, 4, 4)
          
        
        x = self.norm1(x)
        x = self.ct1(x)
        x = F.gelu(x)
        
        x = self.norm2(x)
        x = self.ct2(x)
        x = F.gelu(x)

        x = self.norm3(x)
        x = self.ct3(x)
        x = F.gelu(x)

        x = self.norm4(x)
        x = self.ct4(x)
        x = F.tanh(x)

        
        
        return x