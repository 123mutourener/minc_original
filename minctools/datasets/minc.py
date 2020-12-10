import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class MINC(Dataset):
    def __init__(self, root_dir, set_type='train', classes=range(23),
                 scale=0.233, transform=None):
        self.root_dir = os.path.join(root_dir, "minc")
        self.set_type = set_type
        self.transform = transform
        self.scale = scale
        # This value has been obtained from the MINC paper
        self.mean = torch.Tensor([124 / 255., 117 / 255., 104 / 255.])
        self.std = torch.Tensor([1, 1, 1])

        # I exploit the fact that several patches are obtained from the
        # same image by saving the last used image and by reusing it
        # whenever possible
        self.last_img = dict()
        self.last_img["img_path"] = ''

        # Get the material categories from the categories.txt file
        file_name = os.path.join(self.root_dir, 'categories.txt')
        self.categories = dict()
        new_class_id = 0
        with open(file_name, 'r') as f:
            for class_id, class_name in enumerate(f):
                if class_id in classes:
                    # The last line char (\n) must be removed
                    self.categories[class_id] = [class_name[:-1], new_class_id]
                    new_class_id += 1

        # Load the image data
        set_types = ['train', 'validate', 'test']
        self.data = []
        if set_type == "train":
            set_num = range(1)
        elif set_type == "validate":
            set_num = range(1, 2)
        elif set_type == "test":
            set_num = range(2, 3)
        elif set_type == "all":
            set_num = range(3)
        else:
            raise RuntimeError("invalid data category")

        for i in set_num:
            file_name = os.path.join(self.root_dir, set_types[i] + '.txt')
            with open(file_name, 'r') as f:
                for line in f:
                    # Each row in self.data is composed by:
                    # [label, img_id, patch_center]
                    tmp = line.split(',')
                    label = int(tmp[0])
                    # Check if the patch label is in the new class set
                    if label in self.categories:
                        img_id = tmp[1]
                        patch_x = float(tmp[2])
                        # The last line char (\n) must be removed
                        patch_y = float(tmp[3][:-1])
                        path = os.path.join(self.root_dir, 'photo_orig',
                                            img_id[-1], img_id + '.jpg')
                        patch_center = [patch_x, patch_y]
                        self.data.append([self.categories[label][1], path,
                                          patch_center])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx][1]
        # print(idx)
        if self.last_img["img_path"] != img_path:
            # Sometimes the images are opened as grayscale,
            # so I need to force RGB
            self.last_img["image"] = Image.open(img_path).convert('RGB')
            self.last_img["img_path"] = img_path

        width, height = self.last_img["image"].size
        patch_center = self.data[idx][2]
        patch_center = [patch_center[0] * width,
                        patch_center[1] * height]
        if width < height:
            patch_size = int(width * self.scale)
        else:
            patch_size = int(height * self.scale)
        box = (patch_center[0] - patch_size / 2,
               patch_center[1] - patch_size / 2,
               patch_center[0] + patch_size / 2,
               patch_center[1] + patch_size / 2)
        patch = self.last_img["image"].crop(box)
        if self.transform:
            patch = self.transform(patch)

        # subtract the mean for each channel
        patch = transforms.Compose([
            transforms.Normalize(self.mean, self.std)
        ])(patch)

        label = self.data[idx][0]

        return patch, label
