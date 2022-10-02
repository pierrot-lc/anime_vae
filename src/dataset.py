import os

from PIL import Image
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms


class AnimeDataset(Dataset):
    """Load and transform the images.
    """
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


def load_datasets(path_dir: str, image_size: int, seed: int) -> tuple[AnimeDataset, AnimeDataset]:
    """Instanciate a training and testing dataset by randomly splitting the images.
    A random horizontal flip is applied to the training images.

    Args
    ----
        path_dir: Path to directory containing all images.
        image_size: All images are resized to [image_size, image_size].
        seed: For reproducibility.

    Returns
    -------
        train: Training dataset.
        test: Testing dataset.
    """
    paths = [
        os.path.join(path_dir, p)
        for p in os.listdir(path_dir)
    ]
    paths_train, paths_test = train_test_split(paths, test_size=0.2, random_state=seed)

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # Pixels are between [0, 1]
        transforms.Lambda(lambda p: 2*p - 1),  # Normalize between [-1, 1]
    ])
    train = AnimeDataset(paths_train, transform)

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),  # Pixels are between [0, 1]
        transforms.Lambda(lambda p: 2*p - 1),  # Normalize between [-1, 1]
    ])
    test = AnimeDataset(paths_test, transform)
    return train, test
