#!/usr/bin/env python3
"""
NAL-Powered Adaptive Learning Rates Example

This example uses the Neural Architecture Lab (NAL) to systematically
compare different adaptive learning rate strategies.
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
from src.structure_net.core.network_factory import create_standard_network
from src.structure_net.evolution.adaptive_learning_rates import create_adaptive_training_loop

# --- NAL Test Function ---

def lr_strategy_experiment(config: Dict[str, Any]) -> Tuple[Any, Dict[str, float]]:
    """
    This is the core test function for evaluating a learning rate strategy.
    """
    device = torch.device(config.get('device', 'cpu'))
    
    # 1. Create Network
    network = create_standard_network(
        architecture=config['architecture'],
        sparsity=config['sparsity'],
        device=str(device)
    )

    # 2. Load Data (using dummy data for this example)
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.randn(100, config['architecture'][0]),
            torch.randint(0, config['architecture'][-1], (100,))
        ),
        batch_size=32
    )
    val_loader = train_loader # Use same data for validation in this simple example

    # 3. Run Training with the specified LR strategy
    trained_network, history = create_adaptive_training_loop(
        network=network,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config['epochs'],
        base_lr=config['base_lr'],
        strategy=config['lr_strategy']
    )
    
    # 4. Collect Metrics
    final_metrics = history[-1] if history else {}
    accuracy = final_metrics.get('val_acc', 0.0)
    
    metrics = {
        'final_accuracy': accuracy,
        'convergence_epochs': len(history),
        'final_loss': final_metrics.get('val_loss', 99.0)
    }
    metrics['primary_metric'] = metrics['final_accuracy']
    
    return trained_network, metrics

# --- Main Execution ---

def main():
    """Configures and runs the NAL experiment for LR strategies."""
    print("🚀 NAL Example: Comparing Adaptive Learning Rate Strategies 🚀")

    # 1. Define the Hypothesis
    lr_hypothesis = Hypothesis(
        id="lr_strategy_comparison",
        name="Compare Adaptive Learning Rate Strategies",
        description="Determine which adaptive learning rate strategy yields the best performance.",
        category=HypothesisCategory.TRAINING,
        question="Which high-level adaptive LR strategy (basic, advanced, comprehensive, ultimate) results in the highest accuracy?",
        prediction="More complex strategies will yield better results, with 'comprehensive' or 'ultimate' performing best.",
        test_function=lr_strategy_experiment,
        parameter_space={
            'lr_strategy': ['basic', 'advanced', 'comprehensive', 'ultimate'],
        },
        control_parameters={
            'architecture': [784, 128, 64, 10],
            'sparsity': 0.05,
            'epochs': 10, # Short for a quick example
            'base_lr': 0.001,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        },
        success_metrics={'final_accuracy': 0.1} # Low bar for a demo
    )

    # 2. Configure and run the Lab
    lab_config = LabConfig(
        max_parallel_experiments=4,
        min_experiments_per_hypothesis=4, # Run each strategy once
        results_dir=f"nal_lr_strategy_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    lab = NeuralArchitectureLab(lab_config)
    lab.register_hypothesis(lr_hypothesis)

    # 3. Run the experiment
    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(lab.run_all_hypotheses())
    
    # 4. Print results
    result = results.get("lr_strategy_comparison")
    if result:
        print("\n" + "="*60)
        print("Learning Rate Strategy Comparison Results")
        print("="*60)
        
        # Create a simple ranking
        strategy_performance = []
        for exp_res in result.experiment_results:
            strategy = exp_res.experiment.parameters['lr_strategy']
            accuracy = exp_res.metrics.get('final_accuracy', 0)
            strategy_performance.append({'strategy': strategy, 'accuracy': accuracy})
            
        # Sort by accuracy
        strategy_performance.sort(key=lambda x: x['accuracy'], reverse=True)
        
        print("🏆 Performance Ranking:")
        for i, res in enumerate(strategy_performance):
            print(f"  {i+1}. Strategy: {res['strategy']:<15} | Accuracy: {res['accuracy']:.2%}")

        print("\nKey Insights:")
        for insight in result.key_insights:
            print(f"- {insight}")
    else:
        print("Experiment did not produce a result.")

if __name__ == "__main__":
    main()
