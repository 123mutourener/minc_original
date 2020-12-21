from pytorch_lightning.callbacks import ModelCheckpoint
import os

folder = "checkpoints"


def valid_loss_callback(tag):
    return ModelCheckpoint(
        monitor="valid_loss",
        dirpath=os.path.join(os.getcwd(), folder, tag, "loss"),
        filename="minc-{epoch:02d}-{valid_loss:.2f}",
        save_top_k=5,
        mode="min"
    )


def valid_acc_callback(tag):
    return ModelCheckpoint(
        monitor="valid_accuracy",
        dirpath=os.path.join(os.getcwd(), folder, tag, "accuracy"),
        filename="minc-{epoch:02d}-{valid_accuracy:.2f}",
        save_top_k=5,
        mode="max"
    )


def last_callback(tag):
    return ModelCheckpoint(
        save_last=True,
        dirpath=os.path.join(os.getcwd(), folder, tag, "last"),
        filename= "{epoch:02d}"
    )