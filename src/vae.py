"""
Inspired from https://debuggercafe.com/getting-started-with-variational-autoencoder-using-pytorch/.
"""
from typing import Optional

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange


class ResBlock(nn.Module):
    """A simple residual block.
    """
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
        """Residual forward.
        Does not modify the shape of the input.

        Args
        ----
            x: Input tensor.
                Shape of [batch_size, n_filters, width, height].

        Returns
        -------
            y: Output tensor.
                Shape of [batch_size, n_filters, width, height].
        """
        return self.conv_block(x) + x


class ReduceBlock(nn.Module):
    """Block reducing or upscaling the dimensions of the input tensor.
    """
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
        """Reduce or upscale the dimensions of the input tensor.

        Args
        ----
            x: Input tensor.
                Shape of [batch_size, n_filters_in, width, height].

        Returns
        -------
            y: Output tensor.
                Shape of [batch_size, n_filters_out, width/2, height/2] if `upscale`.
                Shape of [batch_size, n_filters_out, width*2, height*2] if not `upscale`.
        """
        return self.conv_block(x)


class CNNEncoder(nn.Module):
    """Project the input image to an embedding space.
    """
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
        """Project the input tensor.

        Args
        ----
            x: Input tensor.
                Shape of [batch_size, n_channels, width, height].

        Returns
        -------
            x: Projected tensor.
                Shape of [batch_size, n_filters << n_layers, width >> n_layers, height >> n_layers].
        """
        x = self.project_layer(x)
        for layer in self.layers:
            x = layer(x)
        return x


class VAEEncoder(nn.Module):
    """Project the input image to a parameterized latent space.
    """
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
        """Project the input images to a parameterized latent space.

        Args
        ----
            x: Input images.
                Shape of [batch_size, n_channels, width, height].

        Returns
        -------
            x: Parameterized latent images.
                The parameters are the `mu` and `log_var` parameters of a gaussian
                distribution. They are separated along the dimension 1.
                Shape of [batch_size, 2, n_channels_latent, width >> n_layers, height >> n_layers].
        """
        x = self.cnn_encoder(x)
        x = self.project_latent(x)
        return x


class CNNDecoder(nn.Module):
    """Project back the input tensor from the latent space to its original space.
    The RGB channels are not recovered in this module.
    """
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
        """Project back the input tensor to its original shape.

        Args
        ----
            x: Input embedded tensor.
                Shape of [batch_size, n_channels_latent, width, height].

        Returns
        -------
            x: Input tensor projected to its original shape.
                Shape of [batch_size, n_filters, width << n_layers, height << n_layers].
        """
        x = self.project_layer(x)
        for layer in self.layers:
            x = layer(x)
        return x


class VAEDecoder(nn.Module):
    """Project back the input images to the RGB space.
    """
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
        """Project back the input images to the RGB space.

        Args
        ----
            x: Input embedded tensor.
                Shape of [batch_size, n_channels_latent, width, height].

        Returns
        -------
            x: Input tensor projected to its original shape.
                Shape of [batch_size, n_channels, width << n_layers, height << n_layers].
        """
        x = self.cnn_decoder(x)
        x = self.project_rgb(x)
        return x


class VAE(nn.Module):
    """Project the input image to a parameterized latent space
    and project it back to its original RGB space.
    """
    def __init__(
        self,
        n_channels: int,
        n_filters: int,
        n_layers: int,
        n_channels_latent: int,
    ):
        super().__init__()
        self.n_filters = n_filters
        self.n_layers = n_layers
        self.n_channels_latent = n_channels_latent

        self.encoder = VAEEncoder(n_channels, n_filters, n_layers, n_channels_latent)
        self.decoder = VAEDecoder(n_channels, n_filters, n_layers, n_channels_latent)

    def reparameterize(
        self,
        mu: torch.Tensor,
        log_var: torch.Tensor,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """Reparameterization trick.
        Sample from the gaussian distribution parameterized by `mu` and `log_var`.
        To keep the gradient flowing, we only sample a random noise and combine the parameters
        together to simulate a sampling.

        Args
        ----
            mu: Mean of the gaussian.
                Shape of [batch_size, n_channels_latent, width, height].
            log_var: LogVar of the gaussian.
                Shape of [batch_size, n_channels_latent, width, height].

        Returns
        -------
            sample: Simulated sample from the gaussian distribution that keeps the gradient flowing.
        """
        generator = None
        if seed:
            generator = torch.Generator(device=mu.device)
            generator.manual_seed(seed)
        std = torch.exp(0.5*log_var)  # Standard deviation
        eps = torch.randn(std.shape, generator=generator, device=mu.device)  # Small noise to simulate the sampling
        sample = mu + (eps * std)  # Sampling as if coming from the input space
        return sample

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the latent parameters of the input images, and tries to recreate them
        from their latent space.

        Args
        ----
            x: Input images.
                Shape of [batch_size, n_channels, width, height].

        Returns
        -------
            x: Output images.
                Shape of [batch_size, n_channels, width, height].
            mu: Mean of the gaussian.
                Shape of [batch_size, n_channels_latent, width, height].
            log_var: LogVar of the gaussian.
                Shape of [batch_size, n_channels_latent, width, height].
        """
        x = self.encoder(x)

        # Get `mu` and `log_var` from the encoder's output
        mu = x[:, 0]
        log_var = x[:, 1]
        # Sample from the latent space using the reparameterization trick
        z = self.reparameterize(mu, log_var)

        x = self.decoder(z)
        return x, mu, log_var

    @torch.no_grad()
    def generate(self, batch_size: int, image_size: int, seed: Optional[int]=None) -> torch.Tensor:
        """Generate a sample of images from the latent space.

        Args
        ----
            batch_size: Number of images to create.
            image_size: Width and height of the images.
            seed: Seed for experience replay.

        Returns
        -------
            sample: Sampled images from the latent space.
                Shape of [batch_size, n_channels, image_size, image_size].
        """
        device = next(self.parameters()).device
        latent_shape = (batch_size, self.n_channels_latent, image_size >> self.n_layers, image_size >> self.n_layers)

        mu = torch.zeros(latent_shape).to(device)
        log_var = torch.zeros(latent_shape).to(device)

        z = self.reparameterize(mu, log_var, seed)
        sample = self.decoder(z)
        sample = torch.sigmoid(sample)
        return sample

    @torch.no_grad()
    def generate_from_z(self, z: torch.Tensor) -> torch.Tensor:
        x = self.decoder(z)
        x = torch.sigmoid(x)
        return x

    @staticmethod
    def load_from_config(config: dict):
        return VAE(
            config['data']['n_channels'],
            config['net_arch']['n_filters'],
            config['net_arch']['n_layers'],
            config['net_arch']['n_channels_latent'],
        )
