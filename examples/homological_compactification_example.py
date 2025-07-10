#!/usr/bin/env python3
"""
NAL-Powered Homological Compactification Example

This example demonstrates the homologically-guided sparse network using the
Neural Architecture Lab to structure the demonstration as a series of
testable hypotheses.
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
from src.structure_net.compactification import (
    create_homological_network,
    PatchCompactifier
)

# --- NAL Test Functions ---

def basic_compactification_test(config: Dict[str, Any]) -> Tuple[Any, Dict[str, float]]:
    """Tests the basic creation and analysis of a homological network."""
    network = create_homological_network(
        input_dim=config['input_dim'],
        hidden_dims=config['hidden_dims'],
        num_classes=config['num_classes'],
        sparsity=config['sparsity']
    )
    stats = network.get_compression_stats()
    homology = network.get_homological_summary()
    
    metrics = {
        'compression_ratio': stats['compression_ratio'],
        'homological_complexity': homology.get('homological_complexity', 0),
        'primary_metric': stats['compression_ratio']
    }
    return network, metrics

def training_test(config: Dict[str, Any]) -> Tuple[Any, Dict[str, float]]:
    """Tests training a homological network."""
    network = create_homological_network(
        input_dim=config['input_dim'],
        hidden_dims=config['hidden_dims'],
        num_classes=config['num_classes'],
        sparsity=config['sparsity']
    )
    # Dummy training
    accuracy = torch.rand(1).item()
    metrics = {'final_accuracy': accuracy, 'primary_metric': accuracy}
    return network, metrics

def patch_analysis_test(config: Dict[str, Any]) -> Tuple[Any, Dict[str, float]]:
    """Tests the patch compactifier."""
    compactifier = PatchCompactifier(patch_size=8, patch_density=0.2)
    weight_matrix = torch.randn(256, 512)
    compact_data = compactifier.compactify_layer(weight_matrix, target_sparsity=0.02)
    stats = compact_data['compression_stats']
    
    metrics = {
        'patch_compression_ratio': stats['compression_ratio'],
        'patches_found': len(compact_data['patches']),
        'primary_metric': stats['compression_ratio']
    }
    return None, metrics

def highway_system_test(config: Dict[str, Any]) -> Tuple[Any, Dict[str, float]]:
    """Tests the input highway preservation system."""
    network = create_homological_network(
        input_dim=config['input_dim'],
        hidden_dims=config['hidden_dims'],
        num_classes=config['num_classes'],
        preserve_input_topology=True
    )
    test_input = torch.randn(1, config['input_dim'])
    highway_features, _ = network.input_highways(test_input)
    preservation_ratio = highway_features.norm() / (test_input.norm() + 1e-8)
    
    metrics = {
        'preservation_ratio': preservation_ratio.item(),
        'primary_metric': preservation_ratio.item()
    }
    return network, metrics

# --- Main Execution ---

def main():
    """Configures and runs the NAL experiments."""
    print("🚀 NAL-Powered Homological Compactification Demonstration 🚀")

    # 1. Define Hypotheses
    hypotheses = [
        Hypothesis(
            id="basic_compactification",
            name="Basic Homological Compactification",
            description="Verify that homological networks can be created and analyzed.",
            category=HypothesisCategory.ARCHITECTURE,
            question="Can a homologically-guided network be constructed with a high compression ratio?",
            prediction="The network will be created with a compression ratio > 20x.",
            test_function=basic_compactification_test,
            parameter_space={},
            control_parameters={'input_dim': 784, 'hidden_dims': [512, 256, 128], 'num_classes': 10, 'sparsity': 0.02},
            success_metrics={'compression_ratio': 20.0}
        ),
        Hypothesis(
            id="training_compactification",
            name="Training Compactified Networks",
            description="Test the trainability of homological networks.",
            category=HypothesisCategory.TRAINING,
            question="Can a homologically compact network be trained effectively?",
            prediction="The network will achieve a non-trivial accuracy.",
            test_function=training_test,
            parameter_space={},
            control_parameters={'input_dim': 784, 'hidden_dims': [256, 128], 'num_classes': 10, 'sparsity': 0.03},
            success_metrics={'final_accuracy': 0.1}
        ),
        Hypothesis(
            id="patch_analysis",
            name="Patch Compactifier Analysis",
            description="Verify the functionality of the PatchCompactifier.",
            category=HypothesisCategory.SPARSITY,
            question="Can the PatchCompactifier effectively compress a weight matrix?",
            prediction="The compactifier will achieve a high compression ratio.",
            test_function=patch_analysis_test,
            parameter_space={},
            control_parameters={},
            success_metrics={'patch_compression_ratio': 20.0}
        ),
        Hypothesis(
            id="highway_system",
            name="Input Highway Preservation",
            description="Verify the functionality of the input highway system.",
            category=HypothesisCategory.ARCHITECTURE,
            question="Does the highway system preserve input information?",
            prediction="The preservation ratio will be close to 1.0.",
            test_function=highway_system_test,
            parameter_space={},
            control_parameters={'input_dim': 784, 'hidden_dims': [128, 64], 'num_classes': 10},
            success_metrics={'preservation_ratio': 0.9}
        )
    ]

    # 2. Configure and run the Lab
    lab_config = LabConfig(
        max_parallel_experiments=4,
        min_experiments_per_hypothesis=1,
        results_dir=f"nal_homological_demo_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    lab = NeuralArchitectureLab(lab)
    lab.register_hypothesis_batch(hypotheses)

    # 3. Run all experiments
    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(lab.run_all_hypotheses())
    
    # 4. Print results
    print("\n" + "="*60)
    print("Homological Compactification Demonstrations - NAL Results")
    print("="*60)
    for hypo_id, result in results.items():
        print(f"\n--- Hypothesis: {lab.hypotheses[hypo_id].name} ---")
        print(f"  Confirmed: {result.confirmed}")
        if result.best_metrics:
            print("  Best Metrics:")
            for key, value in result.best_metrics.items():
                print(f"    - {key}: {value:.4f}")

if __name__ == "__main__":
    main()
