#!/usr/bin/env python3
"""
Comprehensive Example: NAL-Powered Network Growth

This example demonstrates the full capabilities of the Neural Architecture Lab
to orchestrate a comprehensive, multi-stage growth experiment.
"""

import torch
import asyncio
from typing import Dict, Any, Tuple, List

from src.neural_architecture_lab import (
    NeuralArchitectureLab,
    LabConfig,
    Hypothesis,
    HypothesisCategory
)
from src.structure_net.models.modern_multi_scale_network import ModernMultiScaleNetwork
from src.structure_net.evolution.components import (
    ComposableEvolutionSystem,
    NetworkContext,
    StandardExtremaAnalyzer,
    ExtremaGrowthStrategy,
    StandardNetworkTrainer
)

# --- NAL Test Function ---

def comprehensive_growth_experiment(config: Dict[str, Any]) -> Tuple[Any, Dict[str, float]]:
    """
    This function represents a single experiment run, executed by the NAL.
    It simulates a multi-phase growth process.
    """
    device = torch.device(config.get('device', 'cpu'))
    
    # 1. Initial Network
    network = ModernMultiScaleNetwork(
        initial_architecture=config['architecture'],
        initial_sparsity=config['sparsity']
    ).to(device)
    initial_stats = network.get_stats()

    # Dummy data for demonstration
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.randn(100, config['architecture'][0]),
            torch.randint(0, config['architecture'][-1], (100,))
        ),
        batch_size=32
    )

    # 2. Evolution System Setup
    # A system for adding detail (densification)
    densification_system = ComposableEvolutionSystem()
    densification_system.add_component(StandardExtremaAnalyzer(max_batches=2))
    densification_system.add_component(ExtremaGrowthStrategy(add_layer_on_extrema=False, patch_size=10))
    densification_system.add_component(StandardNetworkTrainer(epochs=1))

    # A system for adding depth
    depth_system = ComposableEvolutionSystem()
    depth_system.add_component(StandardExtremaAnalyzer(max_batches=2))
    depth_system.add_component(ExtremaGrowthStrategy(add_layer_on_extrema=True, new_layer_size=config.get('growth_layer_size', 64)))
    depth_system.add_component(StandardNetworkTrainer(epochs=1))

    # 3. Run Growth Phases
    context = NetworkContext(network, train_loader, device)

    # Coarse phase: Densification
    context = densification_system.evolve_network(context, num_iterations=1)
    
    # Medium phase: Add depth
    context = depth_system.evolve_network(context, num_iterations=1)
    
    # Fine phase: Final densification
    context = densification_system.evolve_network(context, num_iterations=1)
    
    final_network = context.network
    final_stats = final_network.get_stats()

    # 4. Collect Metrics
    metrics = {
        'final_accuracy': torch.rand(1).item(), # Dummy accuracy
        'parameters_added': final_stats['total_parameters'] - initial_stats['total_parameters'],
        'connection_increase': final_stats['active_connections'] - initial_stats['active_connections'],
        'depth_increase': final_stats['depth'] - initial_stats['depth'],
    }
    metrics['primary_metric'] = metrics['final_accuracy']

    return final_network, metrics

# --- Main Execution ---

def main():
    """Configures and runs the NAL experiment."""
    print("🚀 Comprehensive Growth Example (NAL-Powered) 🚀")

    # 1. Define the Hypothesis
    growth_hypothesis = Hypothesis(
        id="comprehensive_growth_demo",
        name="Demonstrate Comprehensive Multi-Phase Growth",
        description="Test a multi-phase growth strategy (densify -> deepen -> densify) on various initial architectures.",
        category=HypothesisCategory.GROWTH,
        question="Can a structured, multi-phase growth strategy effectively evolve different initial network architectures?",
        prediction="The strategy will successfully increase the complexity and performance of all tested architectures.",
        test_function=comprehensive_growth_experiment,
        parameter_space={
            'architecture': [
                [784, 64, 10],
                [784, 128, 64, 10],
                [784, 32, 32, 32, 10]
            ],
            'sparsity': [0.02, 0.05],
            'growth_layer_size': [32, 64]
        },
        control_parameters={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
        success_metrics={'final_accuracy': 0.1} # Low bar for a demo
    )

    # 2. Configure and run the Lab
    lab_config = LabConfig(
        max_parallel_experiments=2,
        min_experiments_per_hypothesis=4,
        results_dir="nal_comprehensive_growth_results"
    )
    
    lab = NeuralArchitectureLab(lab_config)
    lab.register_hypothesis(growth_hypothesis)

    # 3. Run the experiment
    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(lab.run_all_hypotheses())

    # 4. Print results
    result = results.get("comprehensive_growth_demo")
    if result:
        print("\n" + "="*60)
        print("Comprehensive Growth Experiment Results")
        print("="*60)
        print(f"Hypothesis Confirmed: {result.confirmed}")
        print(f"Key Insights:")
        for insight in result.key_insights:
            print(f"- {insight}")
        
        best_params = result.best_parameters
        best_metrics = result.best_metrics
        
        print("\n🏆 Best Performing Configuration:")
        print(f"  - Initial Architecture: {best_params.get('architecture')}")
        print(f"  - Initial Sparsity: {best_params.get('sparsity')}")
        print(f"  - Final Accuracy: {best_metrics.get('final_accuracy'):.2%}")
        print(f"  - Parameters Added: {best_metrics.get('parameters_added')}")

if __name__ == "__main__":
    main()
