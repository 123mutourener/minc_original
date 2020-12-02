import argparse
import sys
import platform
from time import strftime

from pytorchtools.model_parser import get_model, PrintNetList
import ast
import torch
import visdom


def main(args):
    # Model and data parameters
    model = args.model
    dataset = args.dataset
    batch_size = args.batch_size
    classes = ast.literal_eval(args.classes)
    gpu = args.gpu
    seed = args.seed

    # Training parameters
    method = args.method
    epochs = args.epochs
    momentum = args.momentum
    w_decay = args.w_decay

    # Learning rate scheduler parameters
    l_rate = args.l_rate
    scheduler = args.lrate_sched
    step_size = args.step_size
    milestones = ast.literal_eval(args.milestones)
    gamma = args.gamma

    # check what to train
    # Todo: edit the code to load trained patch network if "scene" stage is received
    stage = args.stage
    print("Start to train the {} stage".format(stage))

    # Start training from scratch
    if not args.resume and not args.test:
        # Load the network model
        net = get_model(model, len(classes))
        if net is None:
            print("Unknown model name:", model + ".",
                  "Use '--net-list' option",
                  "to check the available network models")
            sys.exit(2)
        else:
            print("Network {} loaded successfully".format(model))

        if gpu > 0:
            net.cuda()
            print("GPU mode enabled with {} chips".format(gpu))
        else:
            print("CPU mode enabled")

        # Initialize the random generator
        torch.manual_seed(seed)
        if gpu > 0:
            torch.cuda.manual_seed_all(seed)

        # Dictionary used to store the training results and metadata
        json_data = {"platform": platform.platform(), "date": strftime("%Y-%m-%d_%H:%M:%S"), "impl": "pytorch",
                     "datasets": dataset, "gpu": gpu, "model": model, "epochs": epochs, "classes": classes,
                     "stage": stage,
                     "train_params": {"method": method,
                                      "batch_size": batch_size,
                                      "seed": seed,
                                      "last_epoch": 0,
                                      "train_time": 0.0}}
        epochs = range(epochs)

        # Optimization method
        if method == "SGD":
            optimizer = torch.optim.SGD(net.parameters(),
                                        lr=l_rate,
                                        momentum=momentum,
                                        weight_decay=w_decay)

        # Learning rate scheduler
        lrate_dict = dict()
        lrate_dict["sched"] = args.lrate_sched
        if args.lrate_sched is not "constant":
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

        # Extract training parameters from the optimizer state
        for t_param in optimizer.state_dict()["param_groups"][0]:
            if t_param is not "params":
                json_data["train_params"][t_param] = \
                    optimizer.state_dict()["param_groups"][0][t_param]

        num_par = 0
        for parameter in net.parameters():
            num_par += parameter.numel()
        json_data["num_params"] = num_par


def init_data_args(parser):
    # Data Options
    data_args = parser.add_argument_group('Data arguments')
    data_args.add_argument('--dataset', metavar='NAME', default='minc2500',
                           choices=['minc2500', 'minc'],
                           help='name of the datasets to be used' +
                                ' (default: minc2500)')
    data_args.add_argument('--data-root', metavar='DIR', help='path to ' +
                                                              'datasets (default: ./$(DATASET)_root)',
                           default='../data/material/MINC/original-paper')
    data_args.add_argument('--save-dir', metavar='DIR', default='./results',
                           help='path to trained models (default: results/)')
    data_args.add_argument('--chk-dir', metavar='DIR', default='./checkpoints',
                           help='path to checkpoints (default: checkpoints/)')
    data_args.add_argument('--workers', metavar='NUM', type=int,
                           default=8, help='number of worker threads for' +
                                           ' the data loader')


def init_model_args(parser):
    # Model Options
    model_args = parser.add_argument_group('Model arguments')
    model_args.add_argument('-m', '--model', metavar='NAME',
                            default='densenet121', type=str,
                            help='name of the pre-trained netwrok model to be used')
    model_args.add_argument('--classes', metavar='LIST',
                            default='[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,' +
                                    '16,17,18,19,20,21,22]',
                            help='indices of the classes to be used for the' +
                                 ' classification')


def init_train_args(parser):
    # Training Options
    train_args = parser.add_argument_group('Training arguments')
    train_args.add_argument('--method', default='SGD', metavar='NAME',
                            help='training method to be used')
    train_args.add_argument('--gpu', type=int, default=torch.cuda.device_count(), metavar='NUM',
                            help='number of GPUs to use')
    train_args.add_argument('--epochs', default=20, type=int, metavar='NUM',
                            help='number of total epochs to run (default: 20)')
    train_args.add_argument('-b', '--batch-size', default=64, type=int,
                            metavar='NUM',
                            help='mini-batch size (default: 64)')
    train_args.add_argument('--momentum', type=float, default=0.9,
                            metavar='NUM', help='Momentum (default: 0.9)')
    train_args.add_argument('--w-decay', type=float, default=1e-4,
                            metavar='NUM', help='weigth decay (default: 1e-4)')
    train_args.add_argument('--seed', type=int, metavar='NUM',
                            default=179424691,
                            help='random seed (default: 179424691)')


def init_lrate_args(parser):
    # Learning Rate Scheduler Options
    lrate_args = parser.add_argument_group('Learning rate arguments')
    lrate_args.add_argument('--l-rate', type=float, default=0.1,
                            metavar='NUM', help='initial learning Rate' +
                                                ' (default: 0.1)')
    lrate_args.add_argument('--lrate-sched', default="multistep",
                            metavar="NAME", help="name of the learning " +
                                                 "rate scheduler (default: constant)",
                            choices=['step', 'multistep', 'exponential',
                                     'constant'])
    lrate_args.add_argument('--milestones', default='[5,10]', metavar='LIST',
                            help='epoch indices for learning rate reduction' +
                                 ' (multistep, default: [5,10])')
    lrate_args.add_argument('--gamma', type=float, default=0.1,
                            metavar='NUM', help='multiplicative factor of ' +
                                                'learning rate decay (default: 0.1)')
    lrate_args.add_argument('--step-size', type=int, default=5,
                            metavar='NUM', help='pediod of learning rate ' +
                                                'decay (step, default: 5)')


def init_control_args(parser):
    # Other Options
    parser.add_argument('--stage', default='patch', choices=['patch', 'scene'],
                        help='train the patch classification or ' +
                             'full scene segmentation task (default: patch)')
    parser.add_argument('--resume', default='', type=str, metavar='JSON_FILE',
                        help='resume the training from the specified JSON ' +
                             'file  (default: none)')
    parser.add_argument('--test', default='', type=str, metavar='JSON_FILE',
                        help='test the network from the specified JSON file')
    parser.add_argument('--net-list', action=PrintNetList,
                        help='Print the list of the available network ' +
                             'architectures')
    args = parser.parse_args()


if __name__ == '__main__':

    # vis = visdom.Visdom()

    parser = argparse.ArgumentParser(description='Train and test a network ' +
                                                 'on the MINC datasets, refer to section 4.1 and 4.2')
    init_data_args(parser)
    init_model_args(parser)
    init_train_args(parser)
    init_lrate_args(parser)
    init_control_args(parser)

    if not args.net_list:
        main(args)
