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
from adversary import GAN
import PIL as Image



random_seed = 783836
torch.manual_seed(random_seed)



def main():
    model = GAN.load_from_checkpoint("saved_models/epoch=99-step=173800.ckpt")

    model.eval()

    z = torch.randn(1, 100)
    z = z.type_as(model.generator.lin1.weight)
    z= model(z).cpu()
    print(z[0])

    transformer = transforms.ToPILImage(mode='RGB')
    z = transformer(z[0])
    z.show()


if __name__ == "__main__":
    main()