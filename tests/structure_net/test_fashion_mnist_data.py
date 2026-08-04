from unittest.mock import patch

import torch
from torch.utils.data import Dataset, Subset

from structure_net.data_factory import FASHION_MNIST_CONFIG, get_dataset_config
from structure_net.data_factory.datasets import FashionMNISTLoader, get_loader


class _FakeFashionMNIST(Dataset):
    def __init__(self, root, train, transform, download):
        self.root = root
        self.train = train
        self.transform = transform
        self.download = download

    def __len__(self):
        return 100

    def __getitem__(self, index):
        return torch.zeros(1, 28, 28), index % 10


def test_fashion_mnist_is_a_registered_drop_in_mnist_shape():
    config = get_dataset_config("fashion_mnist")

    assert config is FASHION_MNIST_CONFIG
    assert config.input_shape == (28, 28)
    assert config.input_size == 784
    assert config.num_classes == 10
    assert len(config.metadata["classes"]) == 10
    assert isinstance(get_loader("fashion_mnist"), FashionMNISTLoader)


def test_fashion_mnist_subset_is_seeded_without_mutating_numpy_global_state():
    loader = FashionMNISTLoader(FASHION_MNIST_CONFIG)

    with patch(
        "structure_net.data_factory.datasets.torchvision.datasets.FashionMNIST",
        _FakeFashionMNIST,
    ):
        first = loader.load(subset_fraction=0.2, seed=17)
        second = loader.load(subset_fraction=0.2, seed=17)

    assert isinstance(first, Subset)
    assert len(first) == 20
    assert first.indices.tolist() == second.indices.tolist()


def test_fashion_mnist_default_transform_has_expected_shape():
    loader = FashionMNISTLoader(FASHION_MNIST_CONFIG)
    output = loader.get_default_transform()(torch.zeros(28, 28).numpy())

    assert output.shape == (1, 28, 28)
    assert torch.isfinite(output).all()
