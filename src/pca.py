from sklearn.decomposition import PCA

import torch

from src.vae import VAE


class PCAComponents:
    @torch.no_grad()
    def compute_components(self, model: VAE, batch: torch.Tensor):
        _, mu, _ = model(batch)
        self.latent_shape = mu.shape[1:]

        mu = mu.flatten(start_dim=1)
        pca = PCA()
        pca.fit(mu.numpy())
        components = pca.components_  # Shape of [batch_size, latent_size]
        self.components = torch.FloatTensor(components)

    def compute_latent(self, weights: torch.Tensor) -> torch.Tensor:
        full_weights = torch.zeros(self.components.shape[0])
        full_weights[:len(weights)] = weights
        full_weights = full_weights.unsqueeze(dim=0)  # [1, batch_size]
        z = full_weights @ self.components  # [1, latent_size]
        z = z.reshape(self.latent_shape).unsqueeze(dim=0)
        return z
