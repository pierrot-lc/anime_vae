"""
Creates a torch dataset to feed the model.
"""
import os

import cv2

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.utils as vutils

class Dataset(torch.utils.data.dataset.Dataset):
    """Torch dataset for the VAE model.
    Use the paths to images to dynamically read
    images.
    Images are normalized between [0, 1].
    """
    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        image = cv2.imread(path, cv2.COLOR_RGBA2RGB)
        if image is None:
            print('ERROR: image is none:', path)
        image = torch.from_numpy(image).float()
        image = image.permute(2, 0, 1)  # n_channels, dim_x, dim_y
        image = self.transform(image) / 255  # Brings images to [0, 1] range
        return image


def create_dataset(path_dir='images'):
    """Save the paths to all images, and instantiate
    a torch dataset."""
    paths = [os.path.join(path_dir, p)
             for p in os.listdir(path_dir)
             if not p.startswith('.ipynb')]

    print(f'{len(paths):,} images.')

    image_size = 64
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
    ])

    return Dataset(paths, transform)
