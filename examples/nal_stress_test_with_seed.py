#!/usr/bin/env python3
"""
NAL-Powered Stress Test with Pre-trained Seed Model

This script uses the Neural Architecture Lab (NAL) framework to run stress
tests on pre-trained models. It demonstrates how to structure a complex
evaluation task as a formal NAL hypothesis.
"""

import os
import sys
import asyncio
import datetime
import torch
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# NAL Imports
from src.neural_architecture_lab import (
    NeuralArchitectureLab,
    LabConfig,
    Hypothesis,
    HypothesisCategory
)

# structure_net Imports
from src.structure_net.core.io_operations import load_model_seed
from src.structure_net.evolution.integrated_growth_system_v2 import IntegratedGrowthSystem
from src.structure_net.evolution.advanced_layers import ThresholdConfig, MetricsConfig
from src.structure_net.core.network_analysis import get_network_stats
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

# --- Helper Functions (from original script) ---

def load_cifar10_data(batch_size=128, subset_size=None):
    """Load CIFAR-10 dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
    
    if subset_size:
        train_dataset = Subset(train_dataset, range(min(subset_size, len(train_dataset))))
        test_dataset = Subset(test_dataset, range(min(subset_size // 5, len(test_dataset))))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader

def evaluate_model(model, test_loader, device):
    """Evaluate model accuracy."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device).view(data.size(0), -1)
            target = target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return correct / total

# --- NAL Test Function ---

def stress_test_experiment(config: Dict[str, Any]) -> Tuple[Any, Dict[str, float]]:
    """
    This is the core test function executed by the NAL runner for each experiment.
    """
    model_path = config['model_path']
    device = config['device']
    
    # 1. Load Model
    model, checkpoint_data = load_model_seed(model_path, device=str(device))
    initial_accuracy = checkpoint_data.get('accuracy', 0.0)
    
    # 2. Load Data
    train_loader, test_loader = load_cifar10_data(subset_size=config.get('subset_size'))
    
    # 3. Initial Evaluation
    current_accuracy = evaluate_model(model, test_loader, device)
    
    # 4. Growth Test
    growth_system = IntegratedGrowthSystem(
        network=model,
        threshold_config=ThresholdConfig(),
        metrics_config=MetricsConfig()
    )
    growth_results = growth_system.grow_network(
        train_loader=train_loader,
        val_loader=test_loader,
        growth_iterations=config['growth_iterations'],
        epochs_per_iteration=5 # Shorter for testing
    )
    accuracy_after_growth = evaluate_model(model, test_loader, device)

    # 5. Collect Metrics
    final_stats = get_network_stats(model)
    metrics = {
        'initial_accuracy': initial_accuracy,
        'current_accuracy': current_accuracy,
        'accuracy_after_growth': accuracy_after_growth,
        'accuracy_improvement': accuracy_after_growth - current_accuracy,
        'growth_events': len(growth_results.get('growth_history', [])),
        'final_parameters': final_stats['total_parameters']
    }
    
    # The primary metric for success
    metrics['primary_metric'] = metrics['accuracy_improvement']
    
    return model, metrics

# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description="Run NAL-powered stress test with a pre-trained seed model")
    parser.add_argument("model_path", type=str, help="Path to the seed model to be tested.")
    parser.add_argument("--growth-iterations", type=int, default=3, help="Number of growth iterations.")
    parser.add_argument("--quick", action="store_true", help="Run a quick version of the test.")
    args = parser.parse_args()

    if not Path(args.model_path).exists():
        print(f"❌ Error: Model file not found at {args.model_path}")
        sys.exit(1)

    print("🚀 NAL-Powered Stress Test with Pre-trained Seed Model 🚀")

    # 1. Define the Hypothesis for the stress test
    stress_test_hypothesis = Hypothesis(
        id="stress_test_seed_model",
        name="Stress Test Seed Model",
        description="Evaluate the growth potential and robustness of a given seed model.",
        category=HypothesisCategory.GROWTH,
        question="Can a given pre-trained sparse model be improved through dynamic growth?",
        prediction="The growth system will increase the model's accuracy by adding targeted connections.",
        test_function=stress_test_experiment,
        parameter_space={
            'model_path': [args.model_path], # The model to test is a parameter
        },
        control_parameters={
            'growth_iterations': args.growth_iterations,
            'subset_size': 5000 if args.quick else None,
        },
        success_metrics={'accuracy_improvement': 0.01} # Success if accuracy improves by at least 1%
    )

    # 2. Configure and run the Neural Architecture Lab
    lab_config = LabConfig(
        max_parallel_experiments=1, # Only one experiment at a time for this script
        results_dir=f"nal_stress_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        save_best_models=True,
        min_experiments_per_hypothesis=1
    )
    
    lab = NeuralArchitectureLab(lab_config)
    lab.register_hypothesis(stress_test_hypothesis)

    # 3. Run the experiment
    # NAL's async functions need an event loop
    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(lab.run_all_hypotheses())
    
    # 4. Print the results
    result = results.get("stress_test_seed_model")
    if result:
        print("\n" + "="*60)
        print("Stress Test Results")
        print("="*60)
        print(f"Hypothesis Confirmed: {result.confirmed}")
        print(f"Key Insights:")
        for insight in result.key_insights:
            print(f"- {insight}")
        
        best_metrics = result.best_metrics
        print("\nMetrics for the best run:")
        print(f"  - Initial Accuracy: {best_metrics.get('initial_accuracy', 0):.2%}")
        print(f"  - Accuracy Before Growth: {best_metrics.get('current_accuracy', 0):.2%}")
        print(f"  - Accuracy After Growth: {best_metrics.get('accuracy_after_growth', 0):.2%}")
        print(f"  - Improvement: {best_metrics.get('accuracy_improvement', 0):.2%}")
        print(f"  - Growth Events: {best_metrics.get('growth_events', 0)}")
    else:
        print("Stress test did not produce a result.")

if __name__ == "__main__":
    main()
