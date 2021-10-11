"""
Evaluates principal components of batches of images.
Each principal components are averaged over batches.

The final PC are saved onto `components.npy`.
"""
import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from VAE import load_model
from read_data import create_dataset


def encode_batch(model, batch):
    """Return the principal components of a batch.
    The batch is encoded using the model.
    Then the latent vectors, defined by the mean of the gaussians,
    are stacked to compute a PCA of the resulting matrix.

    Args
    ----
    model:
        The VAE neural network.
    batch: [batch_size, 3, 64, 64]
        Torch images of size.

    Return
    ------
    Principal components of the batch.
    This is represented by a numpy matrix of shape [batch_size, latent_size].
    """
    with torch.no_grad():
        code = model.encode(batch)[:, 0]  # Get the means
        code = code.numpy()

    pca = PCA()
    pca.fit(code)

    return pca.components_


if __name__ == '__main__':
    config = {
        'nc': 3,
        'nfilters': 64,
        'latent_size': 128,
        'nlayers': 3,
        'res_layers': 6,
    }
    model = load_model(config, 'hopeful_planet_112.pt')

    dataset = create_dataset()
    loader = DataLoader(
        dataset,
        batch_size=512,
        shuffle=True,
        drop_last=True,
    )

    components = []
    for batch in tqdm(loader):
        components.append(encode_batch(model, batch))

    components = np.mean(np.array(components), axis=0)
    np.save('components.npy', components)
