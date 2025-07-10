#!/usr/bin/env python3
"""
NAL-Powered Fiber Bundle Network Example

Demonstrates the fiber bundle neural network's capabilities using the
Neural Architecture Lab to structure the demonstrations as formal experiments.
"""

import torch
import asyncio
from typing import Dict, Any, Tuple

from src.neural_architecture_lab import (
    NeuralArchitectureLab,
    LabConfig,
    Hypothesis,
    HypothesisCategory
)
from src.structure_net.models.fiber_bundle_network import FiberBundleBuilder
from torch.utils.data import DataLoader, TensorDataset

# --- Helper Functions ---

def create_dummy_data(n_samples=100, in_features=128, out_features=10, device='cpu'):
    """Creates a dummy dataset for demonstration."""
    X = torch.randn(n_samples, in_features).to(device)
    y = torch.randint(0, out_features, (n_samples,)).to(device)
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=32)

# --- NAL Test Functions ---

def geometric_analysis_test(config: Dict[str, Any]) -> Tuple[Any, Dict[str, float]]:
    """Test function for geometric analysis."""
    network = FiberBundleBuilder.create_mnist_bundle()
    curvature = network.compute_connection_curvature(layer_idx=0).item()
    holonomy = network.measure_holonomy(torch.randn(50, network.config.fiber_dim))
    
    metrics = {
        'curvature': curvature,
        'holonomy': holonomy,
        'primary_metric': holonomy 
    }
    return network, metrics

def network_growth_test(config: Dict[str, Any]) -> Tuple[Any, Dict[str, float]]:
    """Test function for network growth."""
    network = FiberBundleBuilder.create_cifar10_bundle()
    initial_params = sum(p.numel() for p in network.parameters())
    
    growth_data = {'performance_metrics': {'accuracy': 0.75}}
    network.grow_network(growth_data)
    
    final_params = sum(p.numel() for p in network.parameters())
    params_added = final_params - initial_params
    
    metrics = {
        'parameters_added': params_added,
        'primary_metric': params_added
    }
    return network, metrics

def catastrophe_detection_test(config: Dict[str, Any]) -> Tuple[Any, Dict[str, float]]:
    """Test function for catastrophe detection."""
    network = FiberBundleBuilder.create_mnist_bundle()
    test_inputs = torch.randn(100, network.config.fiber_dim)
    catastrophic_indices = network.detect_catastrophe_points(test_inputs, epsilon=0.01)
    catastrophe_rate = len(catastrophic_indices) / len(test_inputs)
    
    metrics = {
        'catastrophe_rate': catastrophe_rate,
        'primary_metric': catastrophe_rate
    }
    return network, metrics

# --- Main Execution ---

def main():
    """Configures and runs the NAL experiments for the Fiber Bundle model."""
    print("🚀 NAL-Powered Fiber Bundle Network Demonstration 🚀")

    # 1. Define Hypotheses for each demonstration
    hypotheses = [
        Hypothesis(
            id="fiber_bundle_geometric_analysis",
            name="Geometric Analysis of Fiber Bundles",
            description="Demonstrates the computation of geometric properties like curvature and holonomy.",
            category=HypothesisCategory.ARCHITECTURE,
            question="Can we quantify the geometric properties of a fiber bundle network?",
            prediction="Curvature and holonomy will be measurable and non-zero.",
            test_function=geometric_analysis_test,
            parameter_space={},
            control_parameters={},
            success_metrics={'holonomy': 0.0}
        ),
        Hypothesis(
            id="fiber_bundle_growth",
            name="Geometric-Guided Network Growth",
            description="Demonstrates that the network can grow based on geometric principles.",
            category=HypothesisCategory.GROWTH,
            question="Can the network add connections based on its internal geometry?",
            prediction="The growth mechanism will add a non-zero number of parameters.",
            test_function=network_growth_test,
            parameter_space={},
            control_parameters={},
            success_metrics={'parameters_added': 1}
        ),
        Hypothesis(
            id="fiber_bundle_catastrophe_detection",
            name="Catastrophe Point Detection",
            description="Demonstrates the network's ability to identify points of instability.",
            category=HypothesisCategory.REGULARIZATION,
            question="Can the network identify inputs that cause catastrophic shifts in output?",
            prediction="A non-zero number of catastrophic points will be detected.",
            test_function=catastrophe_detection_test,
            parameter_space={},
            control_parameters={},
            success_metrics={'catastrophe_rate': 0.0}
        )
    ]

    # 2. Configure and run the Lab
    lab_config = LabConfig(
        max_parallel_experiments=2,
        min_experiments_per_hypothesis=1,
        results_dir=f"nal_fiber_bundle_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    lab = NeuralArchitectureLab(lab_config)
    lab.register_hypothesis_batch(hypotheses)

    # 3. Run all experiments
    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(lab.run_all_hypotheses())
    
    # 4. Print results
    print("\n" + "="*60)
    print("Fiber Bundle Demonstrations - NAL Results")
    print("="*60)
    for hypo_id, result in results.items():
        print(f"\n--- Hypothesis: {lab.hypotheses[hypo_id].name} ---")
        print(f"  Confirmed: {result.confirmed}")
        if result.best_metrics:
            print("  Best Metrics:")
            for key, value in result.best_metrics.items():
                print(f"    - {key}: {value:.4f}")
    
    print("\n✅ All demonstrations complete.")

if __name__ == "__main__":
    main()
