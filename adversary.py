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

from discriminator import Discriminator
from generator import Generator

# Putting the two classes together in a GAN
class GAN(pl.LightningModule):
    def __init__(self, latent_dim=100, lr=0.00021):
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters()

        self.generator = Generator(latent_dim=self.hparams.latent_dim)
        self.discriminator = Discriminator()

        # random noise
        self.validation_z = torch.randn(6, self.hparams.latent_dim)

    def forward(self, z):
        return self.generator(z)
    
    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)
    
    def training_step(self, batch, batch_idx):
        
        real_imgs = batch
        batch_size = real_imgs.shape[0]

        # Define the optimizers
        opt_g, opt_d = self.optimizers()

        # Define the labels
        y = torch.ones((batch_size, 1))
        y = y.type_as(real_imgs)

        y_fake = torch.zeros(real_imgs.size(0), 1)
        y_fake = y_fake.type_as(real_imgs)

        # sample noise
        z = torch.randn(real_imgs.shape[0], self.hparams.latent_dim)
        z = z.type_as(real_imgs)

        # train the generator: max log(D(G(z)))
        fake_imgs = self.generator(z)
        y_hat = self.discriminator(fake_imgs)

        g_loss = self.adversarial_loss(y_hat, y)
        opt_g.zero_grad()
        self.manual_backward(g_loss)
        opt_g.step()


        
        # train the discriminator: max log(D(x)) + log(1 - D(G(z)))

        # how well can label as real
        y_hat_real = self.discriminator(real_imgs)

        real_loss = self.adversarial_loss(y_hat_real, y)



        # how well can it label as fake
        y_hat_fake = self.discriminator(self(z).detach())

        fake_loss = self.adversarial_loss(y_hat_fake, y_fake)

        d_loss = (real_loss + fake_loss) / 2
        opt_d.zero_grad()
        self.manual_backward(d_loss)
        opt_d.step()

        self.log_dict({"g_loss": g_loss, "d_loss": d_loss}, prog_bar=True)

        



    def configure_optimizers(self):
        lr = self.hparams.lr
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr)
        return opt_g, opt_d
    
    def plot_imgs(self):
        z = self.validation_z.type_as(self.generator.lin1.weight)
        sample_imgs = self(z).cpu()

        print('epoch', self.current_epoch)
        fig = plt.figure()
        for i in range(sample_imgs.size(0)):
            plt.subplot(2, 3, i+1)
            plt.tight_layout()
            plt.imshow(sample_imgs.detach()[i, 0, :, :], interpolation='none')
            plt.title("Generated Data")
            plt.xticks([])
            plt.yticks([])
            plt.axis("off")
        
        plt.savefig('generated_anime/epoch' + str(self.current_epoch) + '.png')
        plt.close()