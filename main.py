import argparse
from model_parser import get_model, PrintNetList
import torch
import visdom

if __name__ == '__main__':

    vis = visdom.Visdom()

    parser = argparse.ArgumentParser(description='Train and test a network ' +
                                                 'on the MINC datasets, refer to section 4.1 and 4.2')
    # Data Options
    data_args = parser.add_argument_group('Data arguments')
    data_args.add_argument('--datasets', metavar='NAME', default='minc2500',
                           choices=['minc2500', 'minc'],
                           help='name of the datasets to be used' +
                           ' (default: minc2500)')
    data_args.add_argument('--data-root', metavar='DIR', help='path to ' +
                           'datasets (default: ./$(DATASET)_root)', default='../data/material/MINC/original-paper')
    data_args.add_argument('--save-dir', metavar='DIR', default='./results',
                           help='path to trained models (default: results/)')
    data_args.add_argument('--chk-dir', metavar='DIR', default='./checkpoints',
                           help='path to checkpoints (default: checkpoints/)')
    data_args.add_argument('--workers', metavar='NUM', type=int,
                           default=8, help='number of worker threads for' +
                           ' the data loader')

    # Model Options
    model_args = parser.add_argument_group('Model arguments')
    model_args.add_argument('-m', '--model', metavar='NAME',
                            default='densenet121', type=str,
                            help='name of the pre-trained netwrok model to be used')
    model_args.add_argument('--classes', metavar='LIST',
                            default='[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,' +
                            '16,17,18,19,20,21,22]',
                            help='indicies of the classes to be used for the' +
                            ' classification')

    # Training Options
    train_args = parser.add_argument_group('Training arguments')
    train_args.add_argument('--method', default='SGD', metavar='NAME',
                            help='training method to be used')
    train_args.add_argument('--gpu', type=int, default=1, metavar='NUM',
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
                            help='epoch indicies for learning rate reduction' +
                            ' (multistep, default: [5,10])')
    lrate_args.add_argument('--gamma', type=float, default=0.1,
                            metavar='NUM', help='multiplicative factor of ' +
                            'learning rate decay (default: 0.1)')
    lrate_args.add_argument('--step-size', type=int, default=5,
                            metavar='NUM', help='pediod of learning rate ' +
                            'decay (step, default: 5)')

    # Other Options
    parser.add_argument('--resume', default='', type=str, metavar='JSON_FILE',
                        help='resume the training from the specified JSON ' +
                        'file  (default: none)')
    parser.add_argument('--test', default='', type=str, metavar='JSON_FILE',
                        help='test the network from the specified JSON file')
    parser.add_argument('--net-list', action=PrintNetList,
                        help='Print the list of the available network ' +
                        'architectures')

    args = parser.parse_args()

    if not args.net_list:
        main()
