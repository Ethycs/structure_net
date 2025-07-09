import pytest
import torch
from src.structure_net.core.network_factory import create_standard_network

@pytest.fixture(scope="session", autouse=True)
def check_gpu_availability():
    """Fails the test suite if a CUDA-enabled GPU is not available."""
    if not torch.cuda.is_available():
        pytest.fail("GPU not available. This test suite requires a CUDA-enabled GPU.")

@pytest.fixture(scope="session")
def device():
    return torch.device("cuda")

@pytest.fixture(scope="module")
def synthetic_data(device):
    X = torch.randn(1000, 784)
    y = ((X[:, :10].sum(dim=1) + (X[:, 10:20]**2).sum(dim=1)) > 0).long()
    y = (y + ((X[:, 20:30].sum(dim=1) > 0).long() * 2)) % 10
    return X.to(device), y.to(device)

@pytest.fixture
def standard_network(device):
    return create_standard_network([784, 256, 128, 10], sparsity=0.01).to(device)
