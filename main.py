import json
import os
import sys
import torch
# import visdom
from torch.autograd import Variable
from torch import nn
from pytorchtools.data_loader import TorchDataLoader
from pytorchtools.progressbar import progress_bar
from pytorchtools.model_parser import ModelParser
from pytorchtools.arg_parser import ArgParser
from pytorchtools.lr_scheduler import LRScheduler
from pytorchtools.json_formatter import JsonFormatter
from pytorchtools.model_optimizer import ModelOptimizer
from pytorchtools.model_saver import ModelSaver
from tensorboardX import SummaryWriter
from time import time


def main(args):
    # Start training from scratch
    if not args.resume and not args.test:
        # Prepare to train the patch CNN.
        # Parse the argements
        json_formatter = JsonFormatter(args)
        json_data = json_formatter.json_data
        train_info = json_formatter.train_info
        # run the following functions in order!
        # if args.stage == "patch":
        model_parger = ModelParser(json_data)
        net = model_parger.prep_model()
        model_optimizer = ModelOptimizer(args, train_info, net)
        optimizer = model_optimizer.prep_optimizer()
        scheduler = LRScheduler(optimizer, json_data, args)
        scheduler.prep_scheduler()

        # Todo: edit the code to load trained patch network if "scene" stage is received

    # Resume from a training checkpoint or test the network
    else:
        # Todo: implement load saved models, separate patch and scene.
        # Resume from a training checkpoint or test the network
        with open(args.resume or args.test, 'r') as f:
            json_data = json.load(f)
        train_info = json_data["train_params"]

        # Load the network model
        model_parser = ModelParser(json_data)
        net = model_parser.prep_model()

        if args.resume:
            # Resume training
            # Load the saved state
            # (in the same directory as the json file)
            chk_dir = os.path.split(args.resume)[0]
            state = torch.load(os.path.join(chk_dir, json_data["state"]))

            # Load the network parameters
            net.load_state_dict(state["params"])

            # load optimizer
            model_optimizer = ModelOptimizer(args, train_info, net)
            optimizer = model_optimizer.load_optimizer(state)

            # Load the learning rate scheduler info
            scheduler = LRScheduler(optimizer, train_info)
            scheduler.load_scheduler()

        else:
            # Test the network
            # Load the saved parameters
            # (in the same directory as the json file)
            res_dir = os.path.split(args.test)[0]
            if "state" in json_data:
                # Test a checkpointed network
                state = torch.load(os.path.join(res_dir, json_data["state"]))
                net.load_state_dict(state["params"])
            else:
                sys.exit("No network parameters found in JSON file")

    # Prepare the dataset
    dataloader = TorchDataLoader(args, json_data)
    train_loader, val_loader = dataloader.train_loader, dataloader.val_loader
    test_loader = dataloader.test_loader

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
    saver = ModelSaver(args)
    writer = SummaryWriter(os.path.join(args.save_dir, args.tag))
    print("Training network started!")
    dataloaders = dict()
    dataloaders["train"] = train_loader
    dataloaders["val"] = val_loader
    train_info = json_data["train_params"]

    # track the best model
    best_acc = 0.0 if train_info["last_epoch"] == 0 else train_info["best_acc"]
    best_loss = -1
    print('-' * 10)

    for epoch in epochs:
        # calculate the per batch time cost
        start_epoch = time()
        if args.debug:
            phases = ["train"]
        else:
            phases = ["train", "val"]
        # Train the Model
        for phase in phases:
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
                epoch_acc = running_corrects.double().item() / sample_counter

                batch_time += time() - start_batch
                progress_bar(i, len(dataloaders[phase]),
                             "Epoch [{}/{}], {} Loss: {:.4f} Acc: {:.4f}".format(epoch + 1,
                                                                                 epochs[-1] + 1,
                                                                                 "Train" if phase == "train"
                                                                                 else "Validation",
                                                                                 epoch_loss,
                                                                                 epoch_acc))



            # Save the checkpoint state
            if phase == "train":
                scheduler.step()
                # save loss and acc
                writer.add_scalar("train_accuracy", epoch_acc, epoch)
                writer.add_scalar("train_loss", epoch_loss, epoch)

            else:
                writer.add_scalar("valid_accuracy", epoch_acc, epoch)
                writer.add_scalar("valid_loss", epoch_loss, epoch)

        print()
        train_info["train_time"] += round(time() - start_epoch, 3)

        if epoch_acc >= best_acc:
            best_acc = epoch_acc
            train_info["best_acc"] = best_acc
            train_info["best_epoch"] = epoch + 1
            saver.save_state(net, optimizer, json_data, epoch + 1, which="best")

        saver.save_state(net, optimizer, json_data, epoch + 1, which="latest")

    time_elapsed = train_info["train_time"]
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(time_elapsed // 3600,
                                                                (time_elapsed % 3600) // 60,
                                                                time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    writer.close()

if __name__ == '__main__':
    args = ArgParser().args
    # vis = visdom.Visdom()

    if not args.net_list:
        main(args)
