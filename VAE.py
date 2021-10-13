"""
Defines the architecture of the VAE model.

Thanks to https://debuggercafe.com/getting-started-with-variational-autoencoder-using-pytorch/.
The loss and the `reparametrize` functions are drawn from this tutorial.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from einops.layers.torch import Rearrange


class EncoderLayer(nn.Module):
    """Defines a ResNet like layers, with batchnorm and leakyReLU.
    Ends with a transformation that reduces the map size
    but increases the number of filters.
    """
    def __init__(self, nfilters, nlayers):
        """
        Args
        ----
        nfilters: Number of filters of the input.
        nlayers: Number of residual layers.
        """
        super().__init__()

        self.res_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(nfilters, nfilters, 3, padding=1, bias=False),
                nn.BatchNorm2d(nfilters),
                nn.LeakyReLU(),
            )
            for _ in range(nlayers)
        ])

        self.reduce = nn.Sequential(
            nn.Conv2d(nfilters, nfilters * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nfilters * 2),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        """Apply a list of residual convolutional layers, and reduces the map size.
        The reduce operation is done with a strided convolution.

        Args
        ----
        x: [batch_size, filter_size, map_size, map_size]
            The input for the residual layers.

        Return
        ------
        Output shape: [batch_size, filter_size * 2, map_size // 2, map_size // 2]
        """
        for layer in self.res_layers:
            x = x + layer(x)
        return self.reduce(x)

class Encoder(nn.Module):
    """Encoder of an image. Outputs the gaussian parameters of each dimensions."""
    def __init__(self, nc, nfilters, latent_size, nlayers, res_layers, input_size=64):
        """
        Args
        ----
        nc: Number of filters of the input (i.e. 3 for RGB).
        nfilters: Number of filters for the first `EncoderLayer`.
            The following `EncoderLayer` have always 2 times more filters
            than the previous ones.
        latent_size: Dimensionality of the code produced by the encoder.
            This dimensionality defines the number of gaussians predicted for an image.
        nlayers: Number of `EncoderLayer`.
        res_layers: Number of residual layers in each `EncoderLayer`.
        input_size: Map size of the input.
        """
        super(Encoder, self).__init__()

        scale_factor = 1 << nlayers
        final_nfilters = nfilters * scale_factor
        final_size = input_size // scale_factor

        self.bottleneck = nn.Sequential(
            nn.Conv2d(nc, nfilters, 3, padding=1, bias=False),
            nn.BatchNorm2d(nfilters),
            nn.LeakyReLU(),
        )

        self.layers = nn.ModuleList([
            EncoderLayer(nfilters * (1 << i), res_layers)
            for i in range(nlayers)
        ])

        self.final = nn.Sequential(
            nn.Flatten(),
            nn.Linear(final_size * final_size * final_nfilters, latent_size * 2, bias=False),
            nn.LayerNorm(latent_size * 2),
            Rearrange('b (a l) -> b a l', a=2),
        )

    def forward(self, x):
        """Encode a batch of images to a batch of gaussian distributions.

        Args
        ----
        x: [batch_size, nc, image_size, image_size]
            The batch of images.

        Return
        ------
        gaussian_distribution: [batch_size, 2, latent_size]
            Each image is mapped to two vectors.
            The first gives the means of the gaussian distribution.
            The second gives the log-deviations of the gaussian distribution.
            The distribution is describing a density of points in a vector space
            of dimension `latent_size`.
        """
        x = self.bottleneck(x)
        for layer in self.layers:
            x = layer(x)
        return self.final(x)


class DecoderLayer(nn.Module):
    """Defines a ResNet like layers, with batchnorm and leakyReLU.
    Starts with a transformation that upscales the map size
    but decreases the number of filters.
    """
    def __init__(self, nfilters, nlayers):
        """
        Args
        ----
        nfilters: Number of filters of the input.
        nlayers: Number of residual layers.
        """
        super().__init__()

        self.upscale = nn.Sequential(
            nn.ConvTranspose2d(nfilters, nfilters // 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nfilters // 2),
            nn.LeakyReLU(),
        )

        self.res_layers = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(nfilters // 2, nfilters // 2, 3, padding=1, bias=False),
                nn.BatchNorm2d(nfilters // 2),
                nn.LeakyReLU(),
            )
            for _ in range(nlayers)
        ])

    def forward(self, x):
        """Upscales the input and apply a list of residual transposed convolutional layers.
        The upscale operation is done with a strided transposed convolution.

        Args
        ----
        x: [batch_size, filter_size, map_size, map_size]
            The input for the residual layers.

        Return
        ------
        Output shape: [batch_size, filter_size // 2, map_size * 2, map_size * 2]
        """
        x = self.upscale(x)
        for layer in self.res_layers:
            x = x + layer(x)
        return x


class Decoder(nn.Module):
    """Decoder of a latent point. Outputs a decoded image."""
    def __init__(self, nc, nfilters, latent_size, nlayers, res_layers, output_size=64):
        """
        Args
        ----
        nc: Number of filters of the image (i.e. 3 for RGB).
        nfilters: Number of filters for the last `DecoderLayer`.
            The previous `DecoderLayer` have always 2 times more filters
            than the following ones.
        latent_size: Dimensionality of the code produced by the encoder.
            This dimensionality defines the number of gaussians predicted for an image.
            The input of the decoder is expected to be sampled from a gaussian distribution.
        nlayers: Number of `DecoderLayer`.
        res_layers: Number of residual layers in each `DecoderLayer`.
        output_size: Size of the produced image.
        """
        super().__init__()

        scale_factor = 1 << nlayers
        start_nfilters = nfilters * scale_factor
        start_size = output_size // scale_factor

        self.start = nn.Sequential(
            nn.Linear(latent_size, start_size * start_size * start_nfilters),
            nn.LayerNorm(start_size * start_size * start_nfilters),
            nn.LeakyReLU(),

            Rearrange('b (f h w) -> b f h w', w=start_size, h=start_size),
        )

        self.layers = nn.ModuleList([
            DecoderLayer(nfilters * (1 << (nlayers - i)), res_layers)
            for i in range(nlayers)
        ])

        self.bottleneck = nn.Sequential(
            nn.ConvTranspose2d(nfilters, nc, 1, bias=True),
        )  # Map the current filters to the right number of output filters

    def forward(self, x):
        """Decode a batch of points to a batch of images.
        The points are supposed to be drawn from the gaussian distributions.

        Args
        ----
        x: [batch_size, latent_size]
            The batch of points.

        Return
        ------
        images: [batch_size, nc, output_size, output_size]
            The decoded images.
        """
        x = self.start(x)
        for layer in self.layers:
            x = layer(x)
        return self.bottleneck(x)


class VAE(nn.Module):
    """A simple Variational Auto-Encoder.
    The encoder and the decoder are built in a symmetric way.
    """
    def __init__(self, nc, nfilters, latent_size, nlayers, res_layers):
        """
        Args
        ----
        nc: Number of channels in the input image (3 for 'RGB').
        nfilters: Number of filters for the encoder and the decoder layers.
            This only defines the first (last) number of filters of the `EncoderLayer` (`DecoderLayer`).
        latent_size: Dimension of the gaussian distribution used for coding the images.
        nlayers: Number of `EncoderLayer` and `DecoderLayer`.
        res_layers: Number of layers in each `EncoderLayer and `DecoderLayer`.
        """
        super(VAE, self).__init__()
        self.latent_size = latent_size

        # Input is nc x 64 x 64
        self.encoder = Encoder(nc, nfilters, latent_size, nlayers, res_layers)

        # Input is latent_size
        self.decoder = Decoder(nc, nfilters, latent_size, nlayers, res_layers)

    def reparameterize(self, mu, log_var):
        """Draw samples from the gaussian distributions.
        The samples are drawn by adding random noises to
        the means.
        The noise is proportional to the variance of each dimension.

        Args
        ----
        mu: [batch_size, latent_size]
            Mean from the encoder's latent space.
        log_var: [batch_size, latent_size]
            Log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var)  # standard deviation
        eps = torch.randn_like(std)  # `randn_like` as we need the same size
        sample = mu + (eps * std)  # sampling as if coming from the input space
        return sample

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        """Encode & decode a batch of images.

        Args
        ----
        x: [batch_size, nc, image_size, image_size]
            The batch of images.

        Return
        ------
        x: [batch_size, nc, image_size, image_size]
            Reconstructed images.
            No sigmoid is applied to those images.
        mu: [batch_size, latent_size]
            Means of the gaussians.
        log_var: [batch_size, latent_size]
            Log-variance of the gaussians.
        """
        # Encoding
        x = self.encode(x)

        # Get `mu` and `log_var`
        mu = x[:, 0] # The first feature values as mean
        log_var = x[:, 1] # The other feature values as variance
        # Get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)

        # Decoding
        x = self.decode(z)
        return x, mu, log_var

    def generate(self, batch_size, device='cpu'):
        """Generate random images from the latent space.
        The points are drawn from a normal distribution.
        """
        self.to(device)
        with torch.no_grad():
            mu = torch.zeros((batch_size, self.latent_size)).to(device)
            log_var = torch.zeros((batch_size, self.latent_size)).to(device)
            z = self.reparameterize(mu, log_var)
            x = torch.sigmoid(self.decode(z))  # Don't forget to apply the sigmoid
        return x


def load_model(config, path_model='vae.pt'):
    """Load the saved model accordingly to the config.
    Config has to be a dictionnary with all the model hyperparameters.
    """
    model = VAE(
        config['nc'],
        config['nfilters'],
        config['latent_size'],
        config['nlayers'],
        config['res_layers']
    )
    model.load_state_dict(torch.load(path_model, map_location='cpu'))
    model.eval()
    return model
