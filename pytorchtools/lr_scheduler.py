import ast
from torch.optim import lr_scheduler


class LRScheduler():
    def __init__(self, optimizer, json_data=None, args=None):
        if args is not None:
            self._args = args
        self._json_data = json_data
        self._optimizer = optimizer
        self._scheduler = None

    @property
    def json_data(self):
        return self._json_data

    def prep_scheduler(self):
        # Learning rate scheduler parameters
        gamma = self._args.gamma
        lrate_mode = self._args.lrate_sched
        step_size = self._args.step_size
        milestones = ast.literal_eval(self._args.milestones)

        # Learning rate scheduler
        if lrate_mode != "constant":
            if lrate_mode == "step":
                self._scheduler = lr_scheduler.StepLR(self._optimizer, step_size, gamma)
            elif lrate_mode == "multistep":
                self._scheduler = lr_scheduler.MultiStepLR(self._optimizer, milestones, gamma)
            elif lrate_mode == "exponential":
                self._scheduler = lr_scheduler.ExponentialLR(self._optimizer, gamma)

        # Prepare to save the parameters
        lrate_dict = dict()
        lrate_dict["sched"] = lrate_mode
        lrate_dict["step_size"] = step_size
        lrate_dict["gamma"] = gamma
        lrate_dict["milestones"] = milestones

        # Insert the lrate dict to json_data
        self._json_data["train_params"]["l_rate"] = lrate_dict

    def load_scheduler(self):
        lrate_dict = self._json_data["l_rate"]
        lrate_mode = lrate_dict["sched"]
        gamma = lrate_dict["gamma"]
        last_epoch = self._json_data["last_epoch"]

        if lrate_mode != "constant":
            if lrate_mode == "step":
                step_size = self._json_data["step_size"]
                self._scheduler = lr_scheduler.StepLR(self._optimizer, step_size, gamma,
                                                      last_epoch)
            elif lrate_mode == "multistep":
                milestones = lrate_dict["milestones"]
                self._scheduler = lr_scheduler.MultiStepLR(self._optimizer, milestones, gamma,
                                                           last_epoch)
            elif lrate_mode == "exponential":
                self._scheduler = lr_scheduler.ExponentialLR(self._optimizer, gamma,
                                                             last_epoch)

    def step(self):
        if self._scheduler is not None:
            self._scheduler.step()
