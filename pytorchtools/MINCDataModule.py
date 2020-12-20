import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

from minctools.datasets.minc import MINC
from pytorchtools.pl_balanced_distributed_samplier import BalancedDistributedSampler


class MINCDataModule(LightningDataModule):
    def __init__(self, data_root, json_data):
        super().__init__()
        self._use_gpu = torch.cuda.is_available()
        self._data_root = data_root
        self._json_data = json_data
        self._dataset = self._json_data["dataset"]
        self._classes = self._json_data["classes"]
        self._batch_size = self._json_data["train_params"]["batch_size"]
        # self._batch_size = 10

    def train_dataloader(self):
        train_trans = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        train_set = MINC(root_dir=self._data_root, set_type='train',
                         classes=self._classes, transform=train_trans)
        print("Training set loaded, with {} samples".format(len(train_set)))
        # sampler = BalancedDistributedSampler(train_set, 46)
        sampler = BalancedDistributedSampler(train_set, 230000)

        return DataLoader(dataset=train_set,
                          batch_size=self._batch_size,
                          pin_memory=self._use_gpu,
                          sampler=sampler)

    def val_dataloader(self):
        val_trans = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
        val_set = MINC(root_dir=self._data_root, set_type='validate',
                       classes=self._classes, transform=val_trans)
        print("Validation set loaded, with {} samples".format(len(val_set)))
        sampler = DistributedSampler(val_set, shuffle=False)
        # sampler = RandomSampler(val_set, replacement=True, num_samples=46)

        return DataLoader(dataset=val_set, batch_size=self._batch_size,
                          shuffle=False, pin_memory=self._use_gpu, sampler=sampler)

    def test_dataloader(self):
        test_trans = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])

        test_set = MINC(root_dir=self._data_root, set_type='test',
                        classes=self._classes, transform=test_trans)
        print("Test set loaded, with {} samples".format(len(test_set)))
        # if torch.cuda.device_count() > 1:
        sampler = DistributedSampler(test_set, shuffle=False)
        # sampler = RandomSampler(test_set, replacement=True, num_samples=46)
        return DataLoader(dataset=test_set, batch_size=self._batch_size,
                          shuffle=False, pin_memory=self._use_gpu, sampler=sampler)
