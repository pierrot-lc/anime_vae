from sklearn.decomposition import PCA

import torch

from src.vae import VAE


class PCAComponents:
    """A simple object that computes the main PCA components of a batch of samples.
    It translate between the CNN's shape latent vector and the flattened components,
    which allows other apps to use the components freely.
    """
    @torch.no_grad()
    def compute_components(self, model: VAE, batch: torch.Tensor):
        """Computes the PCA components and saves them internally.
        It also discover and saves the CNN's latent shape for later reconstruction.
        """
        _, mu, _ = model(batch)
        self.latent_shape = mu.shape[1:]

        mu = mu.flatten(start_dim=1)
        pca = PCA()
        pca.fit(mu.numpy())
        components = pca.components_  # Shape of [batch_size, latent_size]
        self.components = torch.FloatTensor(components)

    def compute_latent(self, weights: torch.Tensor) -> torch.Tensor:
        """Compute the linear combination of the latent components given their corresponding weights.
        All weights do not have to be given. If only a portion of the weights are given, then
        only the first components are used in the weighted linear combination.

        Args
        ----
            weights: Partial weights of the components.
                Shape of [n_weights,] (n_weights can be smaller than n_components).

        Returns
        -------
            z: Latent vector resulting from the linear combination of the components.
                Its shape is the one saved during the call of `compute_components`.
                Shape of [1, *latent_shape] (with the batch dim).
        """
        full_weights = torch.zeros(self.components.shape[0])
        full_weights[:len(weights)] = weights
        full_weights = full_weights.unsqueeze(dim=0)  # [1, batch_size]
        z = full_weights @ self.components  # [1, latent_size]
        z = z.reshape(self.latent_shape).unsqueeze(dim=0)
        return z
