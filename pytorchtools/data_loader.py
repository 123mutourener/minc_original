import os
from torch.utils.data import DataLoader
from torchvision import transforms
from minctools.datasets.minc import MINC


class TorchDataLoader():
    def __init__(self, args, json_data):
        self._args = args
        self._json_data = json_data
        self._dataset = self._json_data["dataset"]
        self._classes = self._json_data["classes"]
        self._batch_size = self._json_data["train_params"]["batch_size"]

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        if self._args.data_root:
            self._data_root = self._args.data_root
        else:
            # Default directory
            self._data_root = os.path.join(os.curdir, self._dataset + "_root")

        # Load the dataset to PyTorch DataLoader objects
        self.prep_dataset()

    def prep_dataset(self):
        # Prepare data structures
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
        test_trans = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])

        # if dataset == "minc2500":
        #     train_set = MINC2500(root_dir=data_root, set_type='train',
        #                          split=1, transform=train_trans)
        #     val_set = MINC2500(root_dir=data_root, set_type='validate',
        #                        split=1, transform=val_trans)
        if self._dataset == "minc":
            train_set = MINC(root_dir=self._data_root, set_type='train',
                             classes=self._classes, transform=train_trans)
            print("Training set loaded, with {} samples".format(len(train_set)))

            val_set = MINC(root_dir=self._data_root, set_type='validate',
                           classes=self._classes, transform=val_trans)
            print("Validation set loaded, with {} samples".format(len(val_set)))

            test_set = MINC(root_dir=self._data_root, set_type='test',
                            classes=self._classes, transform=test_trans)
            print("Test set loaded, with {} samples".format(len(test_set)))

            self.train_loader = DataLoader(dataset=train_set,
                                           batch_size=self._batch_size,
                                           shuffle=True, num_workers=self._args.workers,
                                           pin_memory=(self._args.gpu > 0))
            self.val_loader = DataLoader(dataset=val_set,
                                         batch_size=self._batch_size,
                                         shuffle=False, num_workers=self._args.workers,
                                         pin_memory=(self._args.gpu > 0))
            self.test_loader = DataLoader(dataset=test_set, batch_size=self._batch_size,
                                          shuffle=False, num_workers=self._args.workers,
                                          pin_memory=(self._args.gpu > 0))
