import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
from torch.utils.data import TensorDataset, DataLoader
from src.structure_net.evolution.components import create_standard_evolution_system, NetworkContext
from src.structure_net.core.network_analysis import get_network_stats

def test_training_integration(standard_network, synthetic_data, device):
    X, y = synthetic_data
    dataloader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)
    
    system = create_standard_evolution_system()
    context = NetworkContext(standard_network, dataloader, device)
    
    initial_connections = get_network_stats(context.network)['total_connections']
    
    evolved_context = system.evolve_network(context, num_iterations=3)
    
    final_connections = get_network_stats(evolved_context.network)['total_connections']
    
    assert final_connections > initial_connections
