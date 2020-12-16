import json
import os
import torch


class ModelSaver():
    def __init__(self, args):
        self._chk_dir = os.path.join(args.chk_dir, args.tag)
        if not os.path.exists(self._chk_dir):
            os.makedirs(self._chk_dir)

    def save_state(self, net, optimizer, json_data, epoch, which="latest"):
        """ Saves the training status.

        Parameters:
        net -- Module object containing the network model;
        optimizer -- Optimizer object obtained from torch.optim
        json_data -- Dictionary used to store the training metadata;
        epoch -- Actual training epoch
        dir -- Directory used to save the data
        """

        json_data["train_params"]["last_epoch"] = epoch
        # epoch_str = '_epoch_' + str(epoch)

        f_name = os.path.join(self._chk_dir, json_data["impl"] + "_" +
                              json_data["model"] + "_" +
                              json_data["dataset"] + "_" +
                              which)
        # Save training state
        state = dict()
        state["params"] = net.state_dict()
        state["optim"] = optimizer.state_dict()
        torch.save(state, f_name + '.state')

        # Update train parameters from optimizer state
        for t_param in state["optim"]["param_groups"][0]:
            if t_param is not "params":
                json_data["train_params"][t_param] = \
                    state["optim"]["param_groups"][0][t_param]

        # Save experiment metadata
        json_data['state'] = os.path.split(f_name + '.state')[1]
        with open(f_name + ".json", 'w') as f:
            json.dump(json_data, f)

    # def save_params(self, net, json_data, dir):
    #     """ Saves the parameteres of the trained network.
    #
    #     Parameters:
    #     net -- Module object containing the network model;
    #     json_data -- Dictionary used to store the training metadata;
    #     dir -- Directory used to save the data
    #     """
    #
    #     if "last_epoch" in json_data["train_params"]:
    #         del json_data["train_params"]["last_epoch"]
    #     if "state" in json_data:
    #         del json_data["state"]
    #
    #     f_name = os.path.join(dir, json_data["impl"] + "_" +
    #                           json_data["model"] + "_" +
    #                           json_data["datasets"] + "_" +
    #                           json_data["UUID"])
    #     # Save training state
    #     torch.save(net.state_dict(), f_name + '.state')
    #     # Save experiment metadata
    #     json_data['params'] = os.path.split(f_name + '.params')[1]
    #     with open(f_name + ".json", 'wb') as f:
    #         json.dump(json_data, f)