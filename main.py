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


# For setting the random seed
random_seed = 42
torch.manual_seed(random_seed)

# Sets the Batchsize
BATCH_SIZE=128

# Checks if GPU is avaliable and then checks to see if there are multiple CPU cores
AVAIL_GPUS = min(1, torch.cuda.device_count())
NUM_WORKERS=int(os.cpu_count() / 2)


# Class that handles Data loading
class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir="./data", 
                 batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = transforms.Compose(
            [
                # Transforms the image to a Tensor
                transforms.ToTensor(),
                # Normalizes the data in the Tensor
                # First number is the mean, second number is the standard deviation of the images
                # TODO: How the fuck do I get these numbers?
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

    # Downloads data
    # TODO: What does the train parameter mean?
    # Answer: I believe this just downloads from different sources, one is a training set the other is a testing set.
    def prepare_data(self):
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    
    def setup(self, stage=None):
        # Assign train/val datasets
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

         # Assign test dataset
        if stage == "test" or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)


    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=self.num_workers)
    
# Detective: fake or no fake -> 1 output [0, 1]
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        # Simple CNN
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 1)
  
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # Flatten the tensor so it can be fed into the FC layers
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return torch.sigmoid(x)
    

# Generate Fake Data: output like real data [1, 28, 28] and values -1, 1
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.lin1 = nn.Linear(latent_dim, 7*7*64)  # [n, 256, 7, 7]
        self.ct1 = nn.ConvTranspose2d(64, 32, 4, stride=2) # [n, 64, 16, 16]
        self.ct2 = nn.ConvTranspose2d(32, 16, 4, stride=2) # [n, 16, 34, 34]
        self.conv = nn.Conv2d(16, 1, kernel_size=7)  # [n, 1, 28, 28]
    

    def forward(self, x):
        # Pass latent space input into linear layer and reshape
        x = self.lin1(x)
        x = F.gelu(x)
        x = x.view(-1, 64, 7, 7)  #256
        
        # Upsample (transposed conv) 16x16 (64 feature maps)
        x = self.ct1(x)
        x = F.gelu(x)
        
        # Upsample to 34x34 (16 feature maps)
        x = self.ct2(x)
        x = F.gelu(x)
        
        # Convolution to 28x28 (1 feature map)
        return self.conv(x)
    
# Putting the two classes together in a GAN
class GAN(pl.LightningModule):
    def __init__(self, latent_dim=100, lr=0.0002):
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
        real_imgs, _ = batch
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
            plt.imshow(sample_imgs.detach()[i, 0, :, :], cmap='gray_r', interpolation='none')
            plt.title("Generated Data")
            plt.xticks([])
            plt.yticks([])
            plt.axis("off")
        
        plt.savefig('gelu_attempts/epoch' + str(self.current_epoch) + '.png')
        plt.close()

# Create a callback class that will do stuff during training
class CallBackReporter(pl.Callback):

    def on_train_epoch_end(self, trainer, pl_module):

        #print images from generator
        pl_module.plot_imgs()

        


def main():
    
    # Create the data module
    dm = MNISTDataModule()

    # Create the model
    cain_abel = GAN()

    cain_abel.plot_imgs()

    trainer = pl.Trainer(callbacks=[CallBackReporter()], max_epochs=20, devices=AVAIL_GPUS, accelerator='gpu')
    trainer.fit(cain_abel, dm)


if __name__ == '__main__':
    main()
















































