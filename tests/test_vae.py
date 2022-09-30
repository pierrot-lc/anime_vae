import pytest
import torch

import src.vae as vae

@torch.no_grad()
@pytest.mark.parametrize(
    'n_channels, n_filters, n_layers, im_size',
    [
        (3, 4, 2, 64),
        (3, 3, 4, 64),
    ]
)
def test_cnn_encoder(
    n_channels: int,
    n_filters: int,
    n_layers: int,
    im_size: int
):
    x = torch.randn((1, n_channels, im_size, im_size))
    output_shape = (1, n_filters << n_layers, im_size >> n_layers, im_size >> n_layers)
    module = vae.CNNEncoder(n_channels, n_filters, n_layers)
    y = module(x)
    assert y.shape == output_shape


@torch.no_grad()
@pytest.mark.parametrize(
    'n_channels, n_filters, n_layers, latent_dim, im_size',
    [
        (3, 4, 2, 10, 64),
        (3, 3, 4, 15, 64),
    ]
)
def test_vae_encoder(
    n_channels: int,
    n_filters: int,
    n_layers: int,
    latent_dim: int,
    im_size: int
):
    x = torch.randn((1, n_channels, im_size, im_size))
    output_shape = (1, 2, latent_dim, im_size >> n_layers, im_size >> n_layers)
    module = vae.VAEEncoder(n_channels, n_filters, n_layers, latent_dim)
    y = module(x)
    assert y.shape == output_shape


@torch.no_grad()
@pytest.mark.parametrize(
    'n_channels, n_filters, n_layers, latent_dim, im_size',
    [
        (3, 4, 2, 10, 64),
        (3, 3, 4, 15, 64),
    ]
)
def test_cnn_decoder(
    n_channels: int,
    n_filters: int,
    n_layers: int,
    latent_dim: int,
    im_size: int
):
    x = torch.randn((1, n_channels, im_size, im_size))
    output_shape = ((1, n_filters, im_size, im_size))
    encoder = vae.VAEEncoder(n_channels, n_filters, n_layers, latent_dim)
    y = encoder(x)

    module = vae.CNNDecoder(latent_dim, n_filters, n_layers)
    x = module(y[:, 0])  # Select one of the duplicated dim
    assert x.shape == output_shape


@torch.no_grad()
@pytest.mark.parametrize(
    'n_channels, n_filters, n_layers, latent_dim, im_size',
    [
        (3, 4, 2, 10, 64),
        (3, 3, 4, 15, 64),
    ]
)
def test_vae(
    n_channels: int,
    n_filters: int,
    n_layers: int,
    latent_dim: int,
    im_size: int
):
    x = torch.randn((1, n_channels, im_size, im_size))
    output_shape = (1, n_channels, im_size, im_size)
    params_shape = (1, latent_dim, im_size >> n_layers, im_size >> n_layers)
    module = vae.VAE(n_channels, n_filters, n_layers, latent_dim)
    y = module(x)

    assert y[0].shape == output_shape
    assert y[1].shape == params_shape
    assert y[2].shape == params_shape
