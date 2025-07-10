import pytest
import torch
from src.structure_net.core.network_factory import create_standard_network

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )

@pytest.fixture(scope="session", autouse=True)
def check_gpu_availability(request):
    """Skip GPU tests if CUDA is not available."""
    if request.node.get_closest_marker("gpu"):
        if not torch.cuda.is_available():
            pytest.skip("GPU not available. This test requires a CUDA-enabled GPU.")

@pytest.fixture(scope="session")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture(scope="session")
def device_cpu():
    """CPU device for tests that don't require GPU."""
    return torch.device("cpu")

@pytest.fixture(scope="module")
def synthetic_data(device):
    X = torch.randn(1000, 784)
    y = ((X[:, :10].sum(dim=1) + (X[:, 10:20]**2).sum(dim=1)) > 0).long()
    y = (y + ((X[:, 20:30].sum(dim=1) > 0).long() * 2)) % 10
    return X.to(device), y.to(device)

@pytest.fixture
def standard_network(device):
    return create_standard_network([784, 256, 128, 10], sparsity=0.01).to(device)
