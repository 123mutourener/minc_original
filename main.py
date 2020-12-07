import copy
import json
import os
import sys
import platform
import ast
import torch
# import visdom
from torch.autograd import Variable
import torch.optim.lr_scheduler as lr_sched
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from minctools.datasets.minc import MINC
from pytorchtools.progressbar import progress_bar
from pytorchtools.model_parser import get_model
from pytorchtools.arg_parser import ArgParser
from time import strftime, time


def main(args):
    # Start training from scratch
    if not args.resume and not args.test:
        # Prepare to train the patch CNN.
        # run the following functions in order!
        # if args.stage == "patch":
        json_data, net = prep_model(args)
        optimizer = prep_optimizer(args, json_data, net)
        scheduler = prep_scheduler(args, json_data, optimizer)
        train_info = json_data["train_params"]
        # Todo: edit the code to load trained patch network if "scene" stage is received

    # Resume from a training checkpoint or test the network
    else:
        # Todo: implement load saved models, separate patch and scene.
        # Resume from a training checkpoint or test the network
        with open(args.resume or args.test, 'rb') as f:
            json_data = json.load(f)
        train_info = json_data["train_params"]
        torch.manual_seed(json_data["seed"])

        # Load the network model
        net = get_model(json_data["model"], len(json_data["classes"]))

        # always check gpu number rather than trust history
        if args.gpu > 0:
            torch.cuda.manual_seed_all(train_info["seed"])
            net.cuda()
        else:
            torch.manual_seed(train_info["seed"])

        if args.resume:
            # Resume training
            # Load the saved state
            # (in the same directory as the json file)
            chk_dir = os.path.split(args.resume)[0]
            state = torch.load(os.path.join(chk_dir, json_data["state"]))

            # Load the network parameters
            net.load_state_dict(state["params"])

            # load optimizer
            optimizer = load_optimizer(train_info, net, state)

            # Load the learning rate scheduler info
            scheduler = load_scheduler(train_info, optimizer, train_info["last_epoch"])

        else:
            # Test the network
            # Load the saved parameters
            # (in the same directory as the json file)
            res_dir = os.path.split(args.test)[0]
            if "params" in json_data:
                net.load_state_dict(torch.load(os.path.join(res_dir,
                                                            json_data["params"]
                                                            )))
            elif "state" in json_data:
                # Test a checkpointed network
                state = torch.load(os.path.join(res_dir, json_data["state"]))
                net.load_state_dict(state["params"])
            else:
                sys.exit("No network parameters found in JSON file")

    # Prepare the dataset
    train_loader, val_loader = prep_dataset(json_data["dataset"], json_data["classes"], train_info["batch_size"],
                                            test=False)
    test_loader = prep_dataset(json_data["dataset"], json_data["classes"], train_info["batch_size"],
                               test=True)
    if not args.test:
        epochs = range(train_info["last_epoch"], train_info["epochs"])
        if args.gpu > 0:
            criterion = nn.CrossEntropyLoss().cuda()
        else:
            criterion = nn.CrossEntropyLoss()
        # Train the network
        train_model(json_data, net, epochs, scheduler, criterion, optimizer, train_loader, val_loader)

    # test the model
    # print("Testing network...")
    # test(net, test_loader, args, json_data)
    # # Save the trained network parameters and the testing results
    # save_params(net, json_data, args.save_dir)


def train_model(json_data, net, epochs, scheduler, criterion, optimizer, train_loader, val_loader):
    """ Train the network on the whole training set

    Parameters:
    net -- Module object containing the network model;
    train_loader -- DataLoader object for the datasets in use;
    criterion -- Method used to compute the loss;
    optimizer -- Method used to update the network paramets;
    epoch -- actual training epoch;
    epochs -- total training epochs;
    loss_window -- visdom window used to plot the loss;
    """
    # Training loop
    print("Training network started!")
    dataloaders = dict()
    dataloaders["train"] = train_loader
    dataloaders["val"] = val_loader

    # track the best model
    best_model_wts = copy.deepcopy(net.state_dict())
    best_acc = 0.0
    best_loss = -1
    print('-' * 10)

    for epoch in epochs:
        # calculate the per batch time cost
        start_epoch = time()

        # Train the Model
        for phase in ["train", "val"]:
            batch_time = 0.0
            if phase == "train":
                # print_interval = 50
                # Switch to train mode
                net.train()
            else:
                net.eval()

            running_loss = 0.0
            running_corrects = 0
            sample_counter = 0

            for i, (images, labels) in enumerate(dataloaders[phase]):
                start_batch = time()

                if args.gpu > 0:
                    images = Variable(images.cuda(non_blocking=True))
                    labels = Variable(labels.cuda(non_blocking=True))
                else:
                    images = Variable(images)
                    labels = Variable(labels)

                # Forward + Backward + Optimize
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = net(images)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * images.size(0)
                running_corrects += torch.sum(preds == labels.data)
                sample_counter += images.size(0)

                epoch_loss = running_loss / sample_counter
                epoch_acc = running_corrects.double() / sample_counter

                batch_time += time() - start_batch
                progress_bar(i, len(dataloaders[phase]),
                             "Epoch [{}/{}], {} Loss: {:.4f} Acc: {:.4f}".format(epoch + 1,
                                                                                 epochs[-1] + 1,
                                                                                 "Train" if phase == "train" \
                                                                                 else "Validation",
                                                                                 epoch_loss,
                                                                                 epoch_acc))

            if phase == "train":
                scheduler.step()

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(net.state_dict())

        print()

        json_data["train_params"]["train_time"] += round(time() - start_epoch, 3)

    time_elapsed = json_data["train_params"]["train_time"]
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(time_elapsed // 3600,
                                                                (time_elapsed % 3600) // 60,
                                                                time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    net.load_state_dict(best_model_wts)
    # Save the checkpoint state
    # save_state(net, optimizer, json_data, epoch + 1, args.chk_dir)


def prep_dataset(dataset, classes, batch_size, test):
    if args.data_root:
        data_root = args.data_root
    else:
        # Default directory
        data_root = os.path.join(os.curdir, dataset + "_root")

    # Prepare data structures
    if not test:
        # Training phase
        train_trans = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        val_trans = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])

        # if dataset == "minc2500":
        #     train_set = MINC2500(root_dir=data_root, set_type='train',
        #                          split=1, transform=train_trans)
        #     val_set = MINC2500(root_dir=data_root, set_type='validate',
        #                        split=1, transform=val_trans)
        if dataset == "minc":
            train_set = MINC(root_dir=data_root, set_type='train',
                             classes=classes, transform=train_trans)
            print("Training set loaded, with {} samples".format(len(train_set)))

            val_set = MINC(root_dir=data_root, set_type='validate',
                           classes=classes, transform=val_trans)
            print("Validation set loaded, with {} samples".format(len(val_set)))

        train_loader = DataLoader(dataset=train_set,
                                  batch_size=batch_size,
                                  shuffle=True, num_workers=args.workers,
                                  pin_memory=(args.gpu > 0))
        val_loader = DataLoader(dataset=val_set,
                                batch_size=batch_size,
                                shuffle=False, num_workers=args.workers,
                                pin_memory=(args.gpu > 0))

        return train_loader, val_loader

    else:
        # Testing phase
        test_trans = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
        # if dataset == "minc2500":
        #     test_set = MINC2500(root_dir=data_root, set_type='test', split=1,
        #                         transform=test_trans)
        if dataset == "minc":
            test_set = MINC(root_dir=data_root, set_type='test',
                            classes=classes, transform=test_trans)
        test_loader = DataLoader(dataset=test_set, batch_size=batch_size,
                                 shuffle=False, num_workers=args.workers,
                                 pin_memory=(args.gpu > 0))
        return test_loader


def load_scheduler(train_info, optimizer, last_epoch):
    if train_info["l_rate"]["sched"] == "step":
        step_size = train_info["l_rate"]["step_size"]
        gamma = train_info["l_rate"]["gamma"]
        scheduler = lr_sched.StepLR(optimizer, step_size, gamma,
                                    last_epoch)
    elif train_info["l_rate"]["sched"] == "multistep":
        milestones = train_info["l_rate"]["milestones"]
        gamma = train_info["l_rate"]["gamma"]
        scheduler = lr_sched.MultiStepLR(optimizer, milestones, gamma,
                                         last_epoch)
    elif args.lrate_sched == "exponential":
        gamma = train_info["l_rate"]["gamma"]
        scheduler = lr_sched.ExponentialLR(optimizer, gamma,
                                           last_epoch)
    return scheduler


def load_optimizer(train_info, net, state):
    # Load the optimizer state
    method = train_info["method"]
    if method == "SGD":
        optimizer = torch.optim.SGD(net.parameters(),
                                    lr=train_info["initial_lr"])
        optimizer.load_state_dict(state["optim"])
    return optimizer


def prep_model(args):
    # Model and data parameters
    model = args.model
    dataset = args.dataset
    classes = ast.literal_eval(args.classes)
    gpu = args.gpu
    seed = args.seed
    stage = args.stage

    print("Start to train the {} stage".format(stage))

    # Load the network model
    net = get_model(model, len(classes))
    if net is None:
        print("Unknown model name:", model + ".",
              "Use '--net-list' option",
              "to check the available network models")
        sys.exit(2)
    else:
        print("Network {} loaded successfully".format(model))

    # Initialize the random generator
    if gpu > 0:
        net.cuda()
        print("GPU mode enabled with {} chips".format(gpu))
        torch.cuda.manual_seed_all(seed)
    else:
        torch.manual_seed(seed)
        print("CPU mode enabled")

    # Dictionary used to store the training results and metadata
    json_data = {"platform": platform.platform(), "date": strftime("%Y-%m-%d_%H:%M:%S"), "impl": "pytorch",
                 "dataset": dataset, "gpu": gpu, "model": model, "classes": classes, "seed": seed,
                 "stage": stage,
                 }

    return json_data, net


def prep_optimizer(args, json_data, net):
    # Training parameters
    momentum = args.momentum
    w_decay = args.w_decay
    method = args.method
    epochs = args.epochs
    batch_size = args.batch_size
    l_rate = args.l_rate

    json_data["train_params"] = {"method": method,
                                 "epochs": epochs,
                                 "batch_size": batch_size,
                                 "last_epoch": 0,
                                 "train_time": 0.0
                                 }
    # Optimization method
    if method == "SGD":
        optimizer = torch.optim.SGD(net.parameters(),
                                    lr=l_rate,
                                    momentum=momentum,
                                    weight_decay=w_decay)
    # Extract training parameters from the optimizer state
    for t_param in optimizer.state_dict()["param_groups"][0]:
        if t_param != "params":
            json_data["train_params"][t_param] = \
                optimizer.state_dict()["param_groups"][0][t_param]

    # get the number of trainable parameters
    num_par = 0
    for parameter in net.parameters():
        num_par += parameter.numel()
    json_data["num_params"] = num_par

    return optimizer


def prep_scheduler(args, json_data, optimizer):
    # Learning rate scheduler parameters
    step_size = args.step_size
    milestones = ast.literal_eval(args.milestones)
    gamma = args.gamma

    # Learning rate scheduler
    lrate_dict = dict()
    lrate_dict["sched"] = args.lrate_sched
    if args.lrate_sched != "constant":
        if args.lrate_sched == "step":
            lrate_dict["step_size"] = step_size
            lrate_dict["gamma"] = gamma
            scheduler = lr_sched.StepLR(optimizer, step_size, gamma)
        elif args.lrate_sched == "multistep":
            lrate_dict["milestones"] = milestones
            lrate_dict["gamma"] = gamma
            scheduler = lr_sched.MultiStepLR(optimizer, milestones, gamma)
        elif args.lrate_sched == "exponential":
            lrate_dict["gamma"] = gamma
            scheduler = lr_sched.ExponentialLR(optimizer, gamma)
    json_data["train_params"]["l_rate"] = lrate_dict
    return scheduler





if __name__ == '__main__':
    args = ArgParser().args
    # vis = visdom.Visdom()

    if not args.net_list:
        main(args)
