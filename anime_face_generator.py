import os

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
import PIL as Image

import matplotlib.pyplot as plt

import pytorch_lightning as pl
from lightning.pytorch import LightningModule

# Local imports
from discriminator import Discriminator
from generator import Generator
from adversary import GAN
from dataloading import DataHandler, animeDataset



# Varaibles I always want
AVAIL_GPUS = min(1, torch.cuda.device_count())
NUM_WORKERS=int(os.cpu_count() / 2)
BATCH_SIZE=64

# Random Seed
random_seed = 42
torch.manual_seed(random_seed)


# Create a callback class that will do stuff during training
class CallBackReporter(pl.Callback):

    def on_train_epoch_end(self, trainer, pl_module):

        #print images from generator
        pl_module.plot_imgs()






def main():
    
    # Create the data module
    dm = DataHandler(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, data_dir='anime_data/images')

    # Create the model
    cain_abel = GAN()

    cain_abel.plot_imgs()

    trainer = pl.Trainer(callbacks=[CallBackReporter()], max_epochs=100, devices=AVAIL_GPUS, accelerator='gpu')
    trainer.fit(cain_abel, dm)

    
    





if __name__ == "__main__":
    main() 













































