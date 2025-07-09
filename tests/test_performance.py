import torch
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
from src.structure_net.core.network_factory import create_standard_network
from src.structure_net.core.network_analysis import get_network_stats

def test_performance_benchmark(device):
    net = create_standard_network([784, 256, 128, 10], sparsity=0.01).to(device)
    x = torch.randn(100, 784).to(device)
    
    # Warmup
    for _ in range(10):
        _ = net(x)
    
    # Benchmark
    start_time = time.time()
    for _ in range(100):
        _ = net(x)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    elapsed = time.time() - start_time
    throughput = 100 * 100 / elapsed
    print(f"Device: {device}, Throughput: {throughput:.0f} samples/sec")
    assert throughput > 1000 # Assert a baseline performance

def test_scalability_stress(device):
    architectures = [
        ([784, 256, 128, 10], "Small"),
        ([784, 512, 256, 128, 10], "Medium"),
    ]
    
    for arch, name in architectures:
        net = create_standard_network(arch, sparsity=0.01).to(device)
        x = torch.randn(32, arch[0]).to(device)
        output = net(x)
        stats = get_network_stats(net)
        assert stats['total_connections'] > 0
