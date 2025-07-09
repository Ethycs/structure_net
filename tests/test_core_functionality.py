import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
from src.structure_net.evolution.components import (
    create_standard_evolution_system,
    NetworkContext,
    StandardExtremaAnalyzer
)
from src.structure_net.core.network_analysis import get_network_stats

def test_basic_functionality(standard_network, device):
    x = torch.randn(32, 784).to(device)
    output = standard_network(x)
    assert output.shape == (32, 10)

    evo_system = create_standard_evolution_system()
    context = NetworkContext(standard_network, None, device)
    evolved_context = evo_system.evolve_network(context, num_iterations=1)
    assert evolved_context.network(x).shape == (32, 10)

def test_sparse_connectivity(standard_network):
    stats = get_network_stats(standard_network)
    assert "overall_sparsity" in stats
    assert 0.009 < stats["overall_sparsity"] < 0.011

def test_extrema_detection(standard_network, device):
    analyzer = StandardExtremaAnalyzer(max_batches=1)
    
    patterns = {
        'random': torch.randn(32, 784),
        'high_values': torch.randn(32, 784) * 3 + 2,
        'low_values': torch.randn(32, 784) * 0.5 - 1,
    }
    
    for pattern_name, pattern in patterns.items():
        dataset = torch.utils.data.TensorDataset(pattern, torch.zeros(32))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)
        context = NetworkContext(standard_network, dataloader, device)
        analysis_result = analyzer.analyze(context)
        assert "total_extrema" in analysis_result.metrics
