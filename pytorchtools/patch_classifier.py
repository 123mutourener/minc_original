import pytorch_lightning as pl
import torch

from pytorchtools.lr_scheduler import LRScheduler
from pytorchtools.model_parser import ModelParser
from torch.nn import functional as F
from pytorchtools.model_optimizer import ModelOptimizer
from pytorchtools.pl_loss import Loss


class PatchClassifier(pl.LightningModule):
    def __init__(self, json_data):
        super(PatchClassifier, self).__init__()
        self.json_data = json_data
        model_parser = ModelParser(json_data)
        self.net = model_parser.prep_model()

        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()
        self.test_acc = pl.metrics.Accuracy()

        self.train_loss = Loss()
        self.valid_loss = Loss()
        self.test_loss = Loss()

    def forward(self, images):
        return self.net(images)

    def backward(self, loss, optimizer, optimizer_idx):
        loss.backward()

    def training_step(self, batch, batch_idx):
        images, labels = batch
        labels_hat = self(images)
        loss = F.cross_entropy(labels_hat, labels)
        _, preds = torch.max(labels_hat, 1)
        self.train_acc(preds, labels)
        self.train_loss(loss, labels)
        self.log("train_accuracy", self.train_acc, prog_bar=True, on_epoch=True, on_step=True, logger=True)
        self.log("train_loss", self.train_loss, prog_bar=True, on_epoch=True, on_step=True, logger=True)
        # print("forward")
        return loss

    # def training_step_end(self, outputs):
    #     # update and log
    #     self.train_acc(outputs["preds"], outputs["targets"])
    #     self.train_loss(outputs["loss"], outputs["targets"])
    #     self.log("train_accuracy", self.train_acc, prog_bar=True, on_epoch=True, on_step=True, logger=True)
    #     self.log("loss", self.train_loss, prog_bar=True, on_epoch=True, on_step=True, logger=True)
    #     return outputs["loss"]

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        labels_hat = self(images)
        loss = F.cross_entropy(labels_hat, labels)
        _, preds = torch.max(labels_hat, 1)
        self.valid_acc(preds, labels)
        self.valid_loss(loss, labels)
        self.log("valid_accuracy", self.valid_acc, prog_bar=False, on_epoch=True, on_step=False, logger=True, sync_dist=True)
        self.log("valid_loss", self.valid_loss, prog_bar=False, on_epoch=True, on_step=False, logger=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        images, labels = batch
        labels_hat = self(images)
        loss = F.cross_entropy(labels_hat, labels)
        _, preds = torch.max(labels_hat, 1)
        self.test_acc(preds, labels)
        self.test_loss(loss, labels)
        self.log("test_accuracy", self.test_acc, prog_bar=False, on_epoch=True, on_step=False, logger=True, sync_dist=True)
        self.log("test_loss", self.test_loss, prog_bar=False, on_epoch=True, on_step=False, logger=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = ModelOptimizer(self.json_data["train_params"], self.net).prep_optimizer()
        scheduler = LRScheduler(optimizer, self.json_data["train_params"]["lrate"]).prep_scheduler()
        return [optimizer], [scheduler]

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("loss", None)
        return items
