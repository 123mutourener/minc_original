import os

import torch


class ModelCheckpoint:
    def __init__(self,
                 monitor=None,
                 dirpath=".",
                 filename="last",
                 save_top_k=5,
                 mode="min"):
        self.monitor = monitor
        self.dirpath = dirpath
        self.filename = filename
        self.save_top_k = save_top_k
        self.mode = mode
        self.best_value = -1 if self.mode=="max" else float("inf") if self.mode=="min" else 0
        self.trainer = None

    def update(self, value=0):
        if self.mode == "max":
            if value >= self.best_value:
                self.best_value = value
                self.trainer.save_checkpoint(self.dirpath, self.filename, self.mode, value, self.save_top_k)
        elif self.mode == "min":
            if value <= self.best_value:
                self.best_value = value
                self.trainer.save_checkpoint(self.dirpath, self.filename, self.mode, value, self.save_top_k)
        else:
            self.trainer.save_checkpoint(self.dirpath, self.filename, self.mode, value, self.save_top_k)

    def set_trainer(self, trainer):
        self.trainer = trainer

    def get_monitor(self):
        return self.monitor