"""
Example of Modern Multi-Scale Network Evolution

This script demonstrates how to evolve a network in a multi-scale fashion
using the new composable evolution system.
"""
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.structure_net.models.modern_multi_scale_network import ModernMultiScaleNetwork
from src.structure_net.evolution.components import (
    ComposableEvolutionSystem,
    NetworkContext,
    StandardExtremaAnalyzer,
    ExtremaGrowthStrategy,
    StandardNetworkTrainer
)

def create_dummy_data(n_samples, in_features, out_features, device):
    """Creates a dummy dataset for demonstration."""
    X = torch.randn(n_samples, in_features).to(device)
    y = torch.randint(0, out_features, (n_samples,)).to(device)
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=32)

def main():
    print("🚀 Modern Multi-Scale Evolution Example 🚀")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Initialize the network with a "coarse" architecture
    initial_arch = [784, 128, 10]
    network = ModernMultiScaleNetwork(initial_arch, initial_sparsity=0.05).to(device)
    print(f"Phase 1 (Coarse): Initialized with architecture {network.get_current_architecture()}")

    # Create dummy data
    train_loader = create_dummy_data(1000, 784, 10, device)

    # 2. Set up the ComposableEvolutionSystem
    # We can configure different systems for different phases
    
    # A system for adding detail (more connections)
    densification_strategy = ExtremaGrowthStrategy(extrema_threshold=0.3, add_layer_on_extrema=False, patch_size=10)
    densification_system = ComposableEvolutionSystem()
    densification_system.add_component(StandardExtremaAnalyzer(max_batches=5))
    densification_system.add_component(densification_strategy)
    densification_system.add_component(StandardNetworkTrainer(epochs=2))

    # A system for adding depth (new layers)
    depth_strategy = ExtremaGrowthStrategy(extrema_threshold=0.5, add_layer_on_extrema=True, new_layer_size=64)
    depth_system = ComposableEvolutionSystem()
    depth_system.add_component(StandardExtremaAnalyzer(max_batches=5))
    depth_system.add_component(depth_strategy)
    depth_system.add_component(StandardNetworkTrainer(epochs=2))

    # 3. Run the evolution in phases (simulating multi-scale growth)
    
    # --- Coarse Phase: Initial training and densification ---
    print("\n--- Running Coarse Phase: Training and Densification ---")
    context = NetworkContext(network, train_loader, device)
    evolved_context = densification_system.evolve_network(context, num_iterations=2)
    network = evolved_context.network
    print(f"Coarse phase complete. Network stats: {network.get_stats()}")

    # --- Medium Phase: Add depth ---
    print("\n--- Running Medium Phase: Adding Depth ---")
    context = NetworkContext(network, train_loader, device)
    evolved_context = depth_system.evolve_network(context, num_iterations=1)
    network = evolved_context.network
    print(f"Medium phase complete. New architecture: {network.get_current_architecture()}")
    print(f"Network stats: {network.get_stats()}")

    # --- Fine Phase: Final densification and refinement ---
    print("\n--- Running Fine Phase: Final Refinement ---")
    context = NetworkContext(network, train_loader, device)
    evolved_context = densification_system.evolve_network(context, num_iterations=1)
    network = evolved_context.network
    print(f"Fine phase complete. Final network stats: {network.get_stats()}")

    print("\n✅ Modern multi-scale evolution complete!")

if __name__ == "__main__":
    main()
