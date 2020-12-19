from pytorch_lightning.callbacks import ModelCheckpoint


def valid_loss_callback():
    return ModelCheckpoint(
        monitor="valid_loss",
        dirpath="./checkpoints/loss",
        filename="minc-{epoch:02d}-{valid_loss:.2f}",
        save_top_k=5,
        mode="min"
    )


def valid_acc_callback():
    return ModelCheckpoint(
        monitor="valid_accuracy",
        dirpath="./checkpoints/accuracy",
        filename="minc-{epoch:02d}-{valid_accuracy:.2f}",
        save_top_k=5,
        mode="max"
    )
