import torch
from torch.utils.data import Sampler


class PySubsetRandomSampler(Sampler):
    def __init__(self, data_source, num_samples=2):
        super().__init__(data_source)
        self.data_source = data_source
        self._num_samples = num_samples
        self._class_image_idx = self.data_source.class_image_idx
        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError(
                "num_samples should be a positive integer "
                "value, but got num_samples={}".format(self.num_samples)
            )

    @property
    def num_samples(self):
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self):
        classes = len(self._class_image_idx.keys())
        per_class_num = self.num_samples // classes

        idx_list = []
        for label, label_idx in self._class_image_idx.items():
            n = len(label_idx)
            idx_list.extend(list(label_idx[torch.randperm(n, dtype=torch.int64)[: per_class_num].tolist()]))

        return iter(idx_list)

    def __len__(self):
        return self.num_samples
