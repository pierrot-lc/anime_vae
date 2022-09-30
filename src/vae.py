import torch
import torch.nn as nn
from einops.layers.torch import Rearrange


class ResBlock(nn.Module):
    def __init__(self, n_filters: int, kernel_size: int):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(
                n_filters,
                n_filters,
                kernel_size,
                stride=1,
                padding='same',
                bias=False,
            ),
            nn.BatchNorm2d(n_filters),
            nn.LeakyReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_block(x) + x


class ReduceBlock(nn.Module):
    def __init__(
        self,
        n_filters_in: int,
        n_filters_out: int,
        upscale: bool = False,
    ):
        super().__init__()
        conv = nn.ConvTranspose2d if upscale else nn.Conv2d
        self.conv_block = nn.Sequential(
            conv(
                n_filters_in,
                n_filters_out,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(n_filters_out),
            nn.LeakyReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_block(x)


class CNNEncoder(nn.Module):
    def __init__(self, n_channels: int, n_filters: int, n_layers: int):
        super().__init__()

        self.project_layer = nn.Sequential(
            nn.Conv2d(
                n_channels,
                n_filters,
                kernel_size=3,
                padding='same',
                bias=False
            ),
            nn.BatchNorm2d(n_filters),
            nn.LeakyReLU(),
        )

        self.layers = nn.ModuleList([
            nn.Sequential(
                ResBlock(n_filters << i, kernel_size=3),
                ReduceBlock(n_filters << i, n_filters << (i+1))
            )
            for i in range(n_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project_layer(x)
        for layer in self.layers:
            x = layer(x)
        return x


class VAEEncoder(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_filters: int,
        n_layers: int,
        n_channels_latent: int,
    ):
        super().__init__()

        self.cnn_encoder = CNNEncoder(n_channels, n_filters, n_layers)
        self.project_latent = nn.Sequential(
            nn.Conv2d(
                n_filters << n_layers,
                n_channels_latent * 2,
                kernel_size=3,
                padding='same',
                bias=True
            ),
            Rearrange('b (d e) w h -> b d e w h', d=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn_encoder(x)
        x = self.project_latent(x)
        return x


class CNNDecoder(nn.Module):
    def __init__(self, n_channels_latent: int, n_filters: int, n_layers: int):
        super().__init__()
        n_filters_start = n_filters << n_layers

        self.project_layer = nn.Sequential(
            nn.Conv2d(
                n_channels_latent,
                n_filters_start,
                kernel_size=3,
                padding='same',
                bias=False,
            ),
            nn.BatchNorm2d(n_filters_start),
            nn.LeakyReLU(),
        )

        self.layers = nn.ModuleList([
            nn.Sequential(
                ResBlock(n_filters << i, kernel_size=3),
                ReduceBlock(n_filters << i, n_filters << (i-1), upscale=True)
            )
            for i in range(n_layers, 0, -1)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project_layer(x)
        for layer in self.layers:
            x = layer(x)
        return x


class VAEDecoder(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_filters: int,
        n_layers: int,
        n_channels_latent: int,
    ):
        super().__init__()

        self.cnn_decoder = CNNDecoder(n_channels_latent, n_filters, n_layers)
        self.project_rgb = nn.Conv2d(
            n_filters,
            n_channels,
            kernel_size=3,
            padding='same'
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn_decoder(x)
        x = self.project_rgb(x)
        return x


class VAE(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_filters: int,
        n_layers: int,
        n_channels_latent: int,
    ):
        super().__init__()

        self.encoder = VAEEncoder(n_channels, n_filters, n_layers, n_channels_latent)
        self.decoder = VAEDecoder(n_channels, n_filters, n_layers, n_channels_latent)

    def reparameterize(
        self,
        mu: torch.Tensor,
        log_var: torch.Tensor
    ) -> torch.Tensor:
        std = torch.exp(0.5*log_var)  # Standard deviation
        eps = torch.randn_like(std)  # Small noise to simulate the sampling
        sample = mu + (eps * std)  # Sampling as if coming from the input space
        return sample

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.encoder(x)

        # Get `mu` and `log_var` from the encoder's output
        mu = x[:, 0]
        log_var = x[:, 1]
        # Sample from the latent space using the reparameterization trick
        z = self.reparameterize(mu, log_var)

        x = self.decoder(z)
        return x, mu, log_var
