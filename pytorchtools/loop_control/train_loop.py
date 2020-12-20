import os
import pytorch_lightning as pl
import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm


class Trainer:
    def __init__(self,
                 progress_bar_refresh_rate=20,
                 log_every_n_steps=20,
                 flush_logs_every_n_steps=800,
                 max_epochs=20,
                 gpus=None,
                 num_nodes=None,
                 # accelerator='ddp',
                 # replace_sampler_ddp=False,
                 callbacks=None,
                 logger=SummaryWriter(os.path.join("./train_logs")),
                 resume_from_checkpoint=None):
        self.progress_bar_refresh_rate = progress_bar_refresh_rate
        self.log_every_n_steps = log_every_n_steps
        self.flush_logs_every_n_steps = flush_logs_every_n_steps

        self.max_epochs = max_epochs
        self.gpus = gpus
        self.num_nodes = num_nodes
        self.callbacks = callbacks
        self.logger = logger
        self.resume_from_checkpoint = resume_from_checkpoint
        self.model = None
        self.data_module = None
        self.optimizers = None
        self.schedulers = None
        self.last_epoch = 0
        self.current_epoch = 1
        self.global_step = 0

    def fit(self, model: pl.LightningModule, data_module=None):
        self.model = model
        self.model.trainer = self
        self.model.logger = self.logger
        self.data_module = data_module if data_module is not None else self.model
        self.optimizers, self.schedulers = self.model.configure_optimizers()
        if self.resume_from_checkpoint:
            self.load_model()

        self.model.prepare_data()
        self.model.setup(stage=None)
        self.train()

    def load_model(self):
        checkpoint = torch.load(self.resume_from_checkpoint)
        self.last_epoch = checkpoint["last_epoch"]
        self.current_epoch = self.last_epoch + 1
        self.global_step = checkpoint["global_step"]
        self.model.load_state_dict(checkpoint["model_state_dict"])

        for idx, optimizer in enumerate(self.optimizers):
            optimizer.load_state_dict.load_state_dict(["optimizer_state_dict"][idx])

        for idx, scheduler in enumerate(self.schedulers):
            scheduler.load_state_dict.load_state_dict(["scheduler_state_dict"][idx])

    def train(self):
        # print("Training network started!")
        # print("-"*10)
        epochs = range(self.current_epoch, self.max_epochs + 1)

        train_loader = self.data_module.train_dataloader()
        valid_loader = self.data_module.val_dataloader()

        for epoch in epochs:
            self.current_epoch = epoch
            # tqdm progress bar
            train_loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False, miniters=len(train_loader)//self.progress_bar_refresh_rate)
            self.model.current_bar = train_loop
            # Train the Model, with both train and validation
            # Switch to train mode
            self.model.train()

            # train for one epoch
            for batch_idx, batch in train_loop:
                loss = self.model.training_step(batch=batch, batch_idx=batch_idx)

                for idx, optimizer in enumerate(self.optimizers):
                    self.model.optimizer_backward(loss=loss, optimizer=optimizer, optimizer_idx=idx)

                self.global_step += 1
                self.model.on_train_batch_end()
            # tqdm progress bar
            eval_loop = tqdm(enumerate(valid_loader), total=len(valid_loader), leave=False, miniters=len(valid_loader)//self.progress_bar_refresh_rate)
            self.model.current_bar = eval_loop
            # evaluate the network
            self.model.eval()
            with torch.no_grad():
                for batch_idx, batch in eval_loop:
                    self.model.validation_step(batch=batch, batch_idx=batch_idx)
                    self.model.on_validation_batch_end()

            self.last_epoch = epoch
