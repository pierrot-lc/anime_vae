import torch

from src.vae import VAE
from src.pca import PCAComponents


def test_components():
    batch_size = 10
    n_channels = 3
    n_filters = 4
    n_layers = 2
    n_channels_latent = 16
    image_size = 32

    model = VAE(n_channels, n_filters, n_layers, n_channels_latent)
    batch = torch.randn((batch_size, n_channels, image_size, image_size))
    pca = PCAComponents()

    pca.compute_components(model, batch)
    assert pca.components.shape == (batch_size, n_channels_latent * (image_size >> n_layers)**2)
    assert pca.latent_shape == (n_channels_latent, image_size >> n_layers, image_size >> n_layers)


def test_latent():
    batch_size = 10
    n_channels = 3
    n_filters = 4
    n_layers = 2
    n_channels_latent = 16
    image_size = 32

    model = VAE(n_channels, n_filters, n_layers, n_channels_latent)
    batch = torch.randn((batch_size, n_channels, image_size, image_size))

    pca = PCAComponents()
    pca.compute_components(model, batch)

    weights = torch.FloatTensor([1, 2, -1, 3])
    z = pca.compute_latent(weights)
    assert z.shape == (1, *pca.latent_shape)

    model.generate_from_z(z)  # Test a generation
