import os

from PIL import Image

import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms


class AnimeDataset(Dataset):
    def __init__(self, paths: list[str], transform: transforms.Compose):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.FloatTensor:
        # Load the image
        path = self.paths[index]
        image = Image.open(path)
        assert image is not None, f"Image '{path}' couldn't be loaded."

        # Apply transformations and load as a tensor
        image = self.transform(image)
        image = torch.FloatTensor(image)
        return image


def load_dataset(path_dir: str, image_size: int) -> AnimeDataset:
    paths = [
        os.path.join(path_dir, p)
        for p in os.listdir(path_dir)
    ]

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # Pixels are between [0, 1]
        transforms.Lambda(lambda p: 2*p - 1),  # Normalize between [-1, 1]
    ])

    return AnimeDataset(paths, transform)