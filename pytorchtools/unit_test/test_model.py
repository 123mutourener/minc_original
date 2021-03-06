import os

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy
from pytorchtools.callbacks import valid_loss_callback, valid_acc_callback, last_callback


class LitMNIST(pl.LightningModule):

    def __init__(self, data_dir='./', hidden_size=64, learning_rate=2e-4):

        super().__init__()
        self.printauto = True
        # Set our init args as class attributes
        self.data_dir = data_dir
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        # Hardcode some dataset specific attributes
        self.num_classes = 10
        self.dims = (1, 28, 28)
        channels, width, height = self.dims
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # Define PyTorch model
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * width * height, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, self.num_classes)
        )

        self.prog_bar_dict = dict()

    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)

    def optimizer_backward(self, loss, optimizer, optimizer_idx):
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    def training_step(self, batch, batch_idx):
        # print(f'Epoch {self.trainer.current_epoch} / Step {self.trainer.global_step}: lr {self.trainer.optimizers[0].param_groups[0]["lr"]}')
        x, y = batch
        # schr = self.trainer.lr_schedulers[0]["scheduler"]

        logits = self(x)
        loss = F.nll_loss(logits, y)
        # schr.step()
        self.log("train_loss", loss.item(), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        # self.log('val_loss', loss, prog_bar=True)
        # self.log('val_acc', acc, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        self.log("valid_loss", torch.randn(1).item())
        self.log("valid_accuracy", torch.randn(1).item())

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # optimizer = torch.optim.SGD(self.parameters(), lr=100)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2], gamma=.1)
        return [optimizer], [lr_scheduler]
        # return optimizer

    ####################
    # DATA RELATED HOOKS
    ####################

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=32)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=32)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=32)

    def log(self, name=None, value=None, prog_bar=True, logger=True, sync_dist=True):
        if logger:
            self.logger.add_scalar(name, value, self.trainer.global_step)
        if prog_bar:
            self.prog_bar_dict[name] = value

    def on_train_batch_end(self):
        self.update_prog_bar()

    def on_validation_batch_end(self):
        self.update_prog_bar()

    def update_prog_bar(self):
        self.current_bar.set_description(f"Epoch [{self.trainer.current_epoch}/{self.trainer.max_epochs}]")
        self.current_bar.set_postfix(self.prog_bar_dict)
        # self.prog_bar_dict = dict()

    def on_train_epoch_end(self):
        pass

    def scheduler_step(self, scheduler, scheduler_idx):
        scheduler.step()

    def on_epoch_end(self, callbacks):
        for callback in callbacks:
            monitor = callback.get_monitor()
            if monitor is None:
                callback.update()
            else:
                callback.update(value=self.prog_bar_dict[monitor])