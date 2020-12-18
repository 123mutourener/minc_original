import torch


class ModelOptimizer():
    def __init__(self, train_info, net):
        self._train_info = train_info
        self._net = net
        self._optimizer = None

    def prep_optimizer(self):
        # Training parameters
        momentum = self._train_info["momentum"]
        w_decay = self._train_info["w_decay"]
        method = self._train_info["method"]
        l_rate = self._train_info["l_rate"]

        # Optimization method
        if method == "SGD":
            self._optimizer = torch.optim.SGD(self._net.parameters(),
                                              lr=l_rate,
                                              momentum=momentum,
                                              weight_decay=w_decay)

        # Extract training parameters from the optimizer state
        for t_param in self._optimizer.state_dict()["param_groups"][0]:
            if t_param != "params":
                self._train_info[t_param] = \
                    self._optimizer.state_dict()["param_groups"][0][t_param]

        # get the number of trainable parameters
        num_par = 0
        for parameter in self._net.parameters():
            num_par += parameter.numel()
        self._train_info["num_params"] = num_par

        return self.optimizer

    def load_optimizer(self, state):
        # Load the optimizer state
        method = self._train_info["method"]
        if method == "SGD":
            self._optimizer = torch.optim.SGD(self._net.parameters(),
                                        lr=self._train_info["initial_lr"])
            self._optimizer.load_state_dict(state["optim"])
        return self.optimizer

    @property
    def optimizer(self):
        return self._optimizer
