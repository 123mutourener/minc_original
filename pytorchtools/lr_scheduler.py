import ast
from torch.optim import lr_scheduler


class LRScheduler():
    def __init__(self, optimizer, lrate_sched_info):
        self._lrate_sched_info = lrate_sched_info
        self._optimizer = optimizer
        self._scheduler = None

    def prep_scheduler(self):
        # Learning rate scheduler
        lrate_mode = self._lrate_sched_info["lrate_sched_mode"]
        step_size = self._lrate_sched_info["step_size"]
        gamma = self._lrate_sched_info["gamma"]
        milestones = self._lrate_sched_info["milestones"]

        if lrate_mode != "constant":
            if lrate_mode == "step":
                self._scheduler = lr_scheduler.StepLR(self._optimizer, step_size, gamma)
            elif lrate_mode == "multistep":
                self._scheduler = lr_scheduler.MultiStepLR(self._optimizer, milestones, gamma)
            elif lrate_mode == "exponential":
                self._scheduler = lr_scheduler.ExponentialLR(self._optimizer, gamma)

    def load_scheduler(self):
        lrate_mode = self._lrate_sched_info["lrate_sched_mode"]
        gamma = self._lrate_sched_info["gamma"]
        last_epoch = self._lrate_sched_info["last_epoch"]

        if lrate_mode != "constant":
            if lrate_mode == "step":
                step_size = self._lrate_sched_info["step_size"]
                self._scheduler = lr_scheduler.StepLR(self._optimizer, step_size, gamma,
                                                      last_epoch)
            elif lrate_mode == "multistep":
                milestones = self._lrate_sched_info["milestones"]
                self._scheduler = lr_scheduler.MultiStepLR(self._optimizer, milestones, gamma,
                                                           last_epoch)
            elif lrate_mode == "exponential":
                self._scheduler = lr_scheduler.ExponentialLR(self._optimizer, gamma,
                                                             last_epoch)

    def step(self):
        if self._scheduler is not None:
            self._scheduler.step()
