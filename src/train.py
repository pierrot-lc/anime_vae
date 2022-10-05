"""Contains the training object.
We use an object so that we do not have to always pass the arguments in the config dict.
"""
from collections import defaultdict

import yaml
import numpy as np
import wandb as wb
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchinfo import summary

from src.vae import VAE
from src.dataset import load_datasets


class TrainVAE:
    """Training module.
    Prepare, train and save a VAE model. It uses wandb for monitoring and
    it saves the config in the internal __dict__ of the module.
    """
    train: dict
    data: dict
    net_arch: dict
    group: str

    def __init__(self, config: dict):
        self.__dict__ |= config
        self.config = config
        self.prepared = False

    def prepare(self):
        """Instanciate the VAE model and optimizer and load the dataset.
        """
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

    def save_state(self):
        """Save the model and its config in the default location `./models/`.
        """
        torch.save(self.model.state_dict(), './models/vae.pth')
        with open('./models/vae.yaml', 'w') as config_file:
            yaml.dump(self.config, config_file)

        self.prepared = False

    def summary(self):
        """Torchinfo summary and device information.
        """
        n_ticks = 50
        print(f'{"-" * (n_ticks+1)} Summary {"-" * n_ticks}')
        summary(self.model, input_size=self.input_size, depth=4)

        print('\nDevice:', self.device)
        print(f'{"-" * (2 * n_ticks + 10)}')

    def batch_forward(
        self,
        batch: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute the loss for the given batch.

        Args
        ----
            batch: Input batch of images.
                It is assumed that images are in the range [-1, 1].
                Shape of [batch_size, n_channels, width, height].

        Returns
        -------
            log: Dictionnary of the following metrics:
                - BCE: Binary crossentropy for image reconstruction.
                - KLD: KL divergence to constrain the latent distribution to be gaussian.
                - loss: BCE + KLD_weight * KLD.
        """
        log = dict()
        predicted, mu, log_var = self.model(batch)
        batch = (batch + 1) / 2  # Normalize between [0, 1] to be a binary loss target

        log['BCE'] = F.binary_cross_entropy_with_logits(predicted, batch, reduction='mean')
        log['KLD'] = -1/2 * (1 + log_var - mu.pow(2) - log_var.exp()).mean()
        log['loss'] =  log['BCE'] + self.train['KLD_weight'] * log['KLD']
        return log

    def train_epoch(self):
        """Train the model for one epoch.
        """
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
        """Compute metrics of the model on the given dataset.

        Args
        ----
            loader: Dataset to be evaluated.

        Returns
        -------
            logs: Metrics evaluated. Those are the BCE, the KLD,
                the loss and some sample of image reconstruction.
        """
        logs_list = defaultdict(list)
        logs = dict()

        # Evaluate the model mean metrics
        self.model.eval()
        for batch in loader:
            batch = batch.to(self.device)
            for metric_name, value in self.batch_forward(batch).items():
                logs_list[metric_name].append(value.cpu().item())

        for metric_name, values in logs_list.items():
            logs[metric_name] = np.mean(values)

        # Logs some predictions
        real = next(iter(loader))[:8].to(self.device)
        predicted, _, _ = self.model(real)
        real = (real + 1) / 2  # Pixels in [0, 1]
        predicted = torch.sigmoid(predicted)  # Pixels in [0, 1]
        logs['images'] = wb.Image(torch.cat((real, predicted), dim=0))

        return logs

    def launch_training(self):
        """Train the model.
        Has to be called once the `prepare` function has been called.
        """
        if not self.prepared:
            self.prepare()

        model = self.model
        model.to(self.device)

        with wb.init(
            entity='pierrotlc',
            group=self.group,
            project='AnimeVAE',
            config=self.config,
            save_code=True,
        ):
            for _ in tqdm(range(self.train['n_epochs'])):
                logs = dict()
                self.train_epoch()

                train_logs = self.validate(self.train_loader)
                test_logs = self.validate(self.test_loader)

                for l, t in [(train_logs, 'Train'), (test_logs, 'Test')]:
                    for m, v in l.items():
                        logs[f'{t} - {m}'] = v

                logs['Generated images'] = wb.Image(self.model.generate(16, self.data['image_size'], self.train['seed']))
                wb.log(logs)

                self.save_state()
