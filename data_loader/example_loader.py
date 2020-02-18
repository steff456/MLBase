import os
import re
from skimage import io, transform

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets

from dataloader_registry import DATALOADER

# Dataset class
@DATALOADER.register('ExampleDataloader')
class ExampleDataset(Dataset):
    """Load the porosity dataset."""
    def __init__(self, data_path):
        """Initialize the dataloader."""
        super(ExampleDataset, self).__init__()

    def __len__(self):
        """Denote the total number of samples."""
        pass

    def __getitem__(self, index):
        """Generate one sample of data."""
        pass
