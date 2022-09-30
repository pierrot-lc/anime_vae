from collections import defaultdict

import numpy as np
import wandb as wb

import torch
import torch.optim as optim
import torch.nn.functional as F
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
        torch.manual_seed(self.train['seed'])

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

    def batch_forward(
        self,
        batch: torch.Tensor,
    ) -> dict:
        log = dict()
        predicted, mu, log_var = self.model(batch)

        log['BCE'] = F.binary_cross_entropy_with_logits(predicted, batch, reduction='mean')
        log['KLD'] = -1/2 * (1 + log_var - mu.pow(2) - log_var.exp()).mean()
        log['loss'] =  log['BCE'] + self.train['KLD_weight'] * log['KLD']
        return log

    def train_epoch(self):
        self.model.train()
        for batch in self.train_loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            loss = self.batch_forward(batch)['loss']

            loss.backward()
            # nn.utils.clip_grad_norm_(self.model.parameters(), self.train['clip_grad_norm'])
            self.optimizer.step()

    @torch.no_grad()
    def validate(self, loader: DataLoader) -> dict:
        logs_list = defaultdict(list)
        logs = dict()

        # Evaluate the model mean metrics
        self.model.eval()
        for batch in loader:
            for metric_name, value in self.batch_forward(batch).items():
                logs_list[metric_name].append(value)

        for metric_name, values in logs_list.items():
            logs[metric_name] = np.mean(values)

        # Logs some predictions
        real = next(iter(loader))[:8].to(self.device)
        predicted, _, _ = self.model(real)
        predicted = torch.sigmoid(predicted)
        logs['images'] = wb.Image(torch.cat((real, predicted), dim=0))

        return logs

    def launch_training(self):
        if not self.prepared:
            self.prepare()

        model = self.model
        model.to(self.device)

        for _ in range(self.train['n_epochs']):
            logs = dict()
            self.train_epoch()

            train_logs = self.validate(self.train_loader)
            test_logs = self.validate(self.test_loader)

            for l, t in [(train_logs, 'Train'), (test_logs, 'Test')]:
                for m, v in l.items():
                    logs[f'{t} - {m}'] = v

            logs['Generated images'] = wb.Image(self.model.generate(16, self.data['image_size']))

            print('\nEpoch:')
            print(logs['Train - loss'])
            print(logs['Test - loss'])
