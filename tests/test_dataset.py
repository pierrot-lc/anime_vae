import pytest
import numpy as np

from src.dataset import load_datasets


def test_loading():
    im_shape = (3, 32, 32)
    dataset = load_datasets('./images', im_shape[-1], 0)[0]
    for i in range(min(200, len(dataset))):
        image = dataset[i]
        assert (image.abs() <= 1).all()
        assert tuple(image.shape) == im_shape
