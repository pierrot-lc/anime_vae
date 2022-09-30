import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchinfo import summary

from src.vae import VAE
from src.dataset import load_datasets


class TrainVAE:
    train: dict
    data: dict
    net_arch: dict

    def __init__(self, config: dict):
        self.__dict__ |= config
        self.prepared = False

    def prepare(self):
        self.input_size = (
            self.train['batch_size'],
            self.data['n_channels'],
            self.data['image_size'],
            self.data['image_size'],
        )

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Build model
        self.model = VAE(
            self.data['n_channels'],
            self.net_arch['n_filters'],
            self.net_arch['n_layers'],
            self.net_arch['n_channels_latent'],
        )

        # Load data
        train_set, test_set = load_datasets(self.data['path_dir'], self.data['image_size'], self.train['seed'])
        self.train_set = train_set
        self.test_set = test_set
        self.train_loader = DataLoader(train_set, batch_size=self.train['batch_size'], shuffle=True)
        self.test_loader = DataLoader(test_set, batch_size=self.train['batch_size'], shuffle=False)

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.train['lr'])

        self.prepared = True  # Preparation is done

    def summary(self):
        n_ticks = 50
        print(f'{"-" * (n_ticks+1)} Summary {"-" * n_ticks}')
        summary(self.model, input_size=self.input_size)

        print('\nDevice:', self.device)
        print(f'{"-" * (2 * n_ticks + 10)}')

    def launch_training(self):
        if not self.prepared:
            self.prepare()

        model = self.model
        model.to(self.device)
