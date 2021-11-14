import os                       # for working with files
import numpy as np              # for numerical computationss
import pandas as pd             # for working with dataframes
import torch                    # Pytorch module
import matplotlib.pyplot as plt # for plotting informations on graph and images using tensors
import torch.nn as nn           # for creating  neural networks
from torch.utils.data import DataLoader # for dataloaders
from PIL import Image           # for checking images
import torch.nn.functional as F # for functions for calculating loss
import torchvision.transforms as transforms   # for transforming images into tensors
from torchvision.utils import make_grid       # for data checking
from torchvision.datasets import ImageFolder  # for working with classes and images
from torchsummary import summary              # for getting the summary of our model
from pprint import pprint
from tqdm import tqdm

from ribes.utils import to_device, check_kaggle


def get_home():
    if check_kaggle():
        home = "../input"
    else:
        home = "input"
    return home


def read_paths():
    home = get_home()
    data_dir = os.path.join(home,
                            "new-plant-diseases-dataset",
                            "New Plant Diseases Dataset(Augmented)",
                            "New Plant Diseases Dataset(Augmented)")
    assert os.path.isdir(data_dir)
    train_dir = os.path.join(data_dir, "train")
    assert os.path.isdir(data_dir)
    valid_dir = os.path.join(data_dir, "valid")
    assert os.path.isdir(valid_dir)
    return data_dir, train_dir, valid_dir


class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


def get_dataloader(batch_size, device):
    _, train_dir, valid_dir = read_paths()

    train_if = ImageFolder(train_dir, transform=transforms.ToTensor())
    valid_if = ImageFolder(valid_dir, transform=transforms.ToTensor())

    train_dl = DataLoader(train_if, batch_size, shuffle=True, num_workers=2, pin_memory=False)
    valid_dl = DataLoader(valid_if, batch_size, num_workers=2, pin_memory=False)

    train_ddl = DeviceDataLoader(train_dl, device)
    valid_ddl = DeviceDataLoader(valid_dl, device)

    return train_ddl, valid_ddl, train_if, valid_if