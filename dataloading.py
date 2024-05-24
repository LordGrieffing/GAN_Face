import os

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from PIL import Image

import matplotlib.pyplot as plt

import pytorch_lightning as pl
from lightning.pytorch import LightningModule

# Class that handles Data loading
class DataHandler(pl.LightningDataModule):
    def __init__(self, batch_size, num_workers, data_dir):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = transforms.Compose(
            [
                # Need to rescale the images so that they are uniform size
                transforms.Resize((64,64)),

                # Transforms the image to a Tensor
                transforms.ToTensor()
                #transforms.Normalize()
                
            ]
        )

    
    def setup(self, stage=None):
        # Assign train/val datasets
        if stage == "fit" or stage is None:
            anime_full = animeDataset(image_dir= self.data_dir, transform=self.transform)
            self.anime_train, self.anime_val = random_split(anime_full, [55611, 7954])

         # Assign test dataset
        if stage == "test" or stage is None:
            self.anime_test = animeDataset(image_dir= self.data_dir, transform=self.transform)


    def train_dataloader(self):
        return DataLoader(self.anime_train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.anime_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.anime_test, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def display_sample(self):
        return self.anime_train.__getitem__(0)


# For building a custom dataset
class animeDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image