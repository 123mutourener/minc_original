import argparse
import shortuuid
from pytorchtools.model_parser import PrintNetList
from torch.cuda import device_count


class ArgParser:
    def __init__(self):
        parser = argparse.ArgumentParser(description='Train and test a network ' +
                                                     'on the MINC datasets, refer to section 4.1 and 4.2')
        self._init_data_args(parser)
        self._init_model_args(parser)
        self._init_train_args(parser)
        self._init_lrate_args(parser)
        self._init_control_args(parser)

        self._args = parser.parse_args()

    @property
    def args(self):
        return self._args

    @staticmethod
    def _str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    @staticmethod
    def _init_data_args(parser):
        # Data Options
        data_args = parser.add_argument_group('Data arguments')
        data_args.add_argument('--dataset', metavar='NAME', default='minc',
                               choices=['minc2500', 'minc'],
                               help='name of the datasets to be used' +
                                    ' (default: minc2500)')
        data_args.add_argument('--data-root', metavar='DIR', help='path to ' +
                                                                  'datasets (default: ./$(DATASET)_root)',
                               default='../../data/material/MINC/original-paper/')
        data_args.add_argument('--save-dir', metavar='DIR', default='./results',
                               help='path to trained models (default: results/)')
        data_args.add_argument('--chk-dir', metavar='DIR', default='./checkpoints',
                               help='path to checkpoints (default: checkpoints/)')
        data_args.add_argument('--workers', metavar='NUM', type=int,
                               default=0, help='number of worker threads for' +
                                               ' the data loader')

    @staticmethod
    def _init_control_args(parser):
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
        parser.add_argument('--tag', default='random_sampling',
                            help='The reference name of the experiment')

    @staticmethod
    def _init_lrate_args(parser):
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

    def _init_train_args(self, parser):
        # Training Options
        train_args = parser.add_argument_group('Training arguments')
        train_args.add_argument('--method', default='SGD', metavar='NAME',
                                help='training method to be used')
        train_args.add_argument('--gpu', type=int, default=device_count(), metavar='NUM',
                                help='number of GPUs to use')
        train_args.add_argument('--epochs', default=20, type=int, metavar='NUM',
                                help='number of total epochs to run (default: 20)')
        train_args.add_argument('-b', '--batch-size', default=128, type=int,
                                metavar='NUM',
                                help='mini-batch size (default: 64)')
        train_args.add_argument('--momentum', type=float, default=0.9,
                                metavar='NUM', help='Momentum (default: 0.9)')
        train_args.add_argument('--w-decay', type=float, default=1e-4,
                                metavar='NUM', help='weigth decay (default: 1e-4)')
        train_args.add_argument('--seed', type=int, metavar='NUM',
                                default=179424691,
                                help='random seed (default: 179424691)')
        train_args.add_argument('--debug', default=False, const=True, metavar='DEBUG', type=self._str2bool, nargs="?",
                                help='enable the debug mode (default: False)')

    @staticmethod
    def _init_model_args(parser):
        # Model Options
        model_args = parser.add_argument_group('Model arguments')
        model_args.add_argument('-m', '--model', metavar='NAME',
                                default='densenet121', type=str,
                                help='name of the pre-trained network model to be used')
        model_args.add_argument('--classes', metavar='LIST',
                                default='[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,' +
                                        '16,17,18,19,20,21,22]',
                                help='indices of the classes to be used for the' +
                                     ' classification')
