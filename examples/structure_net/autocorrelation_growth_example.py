#!/usr/bin/env python3
"""
Example: Using the Autocorrelation Framework for Meta-Learning Growth

This example demonstrates how to use the new MetricPerformanceAnalyzer
to discover which metrics predict learning success and automatically
optimize growth strategies based on learned patterns.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import sys
import os

from src.structure_net.core.network_factory import create_standard_network
from src.structure_net.evolution.components import create_standard_evolution_system, NetworkContext
from src.structure_net.evolution.metrics import CompleteMetricsSystem, ThresholdConfig, MetricsConfig
from src.structure_net.evolution.autocorrelation import PerformanceAnalyzer


def create_sample_data():
    """Create sample MNIST data for demonstration."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Use smaller subset for faster demonstration
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    # Create smaller subsets for faster execution
    train_subset = torch.utils.data.Subset(train_dataset, range(1000))
    test_subset = torch.utils.data.Subset(test_dataset, range(200))
    
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    val_loader = DataLoader(test_subset, batch_size=32, shuffle=False)
    
    return train_loader, val_loader

def main():
    """Demonstrate the autocorrelation framework."""
    print("🚀 Autocorrelation Framework Demo")
    print("="*50)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders
    print("\n📊 Loading MNIST data...")
    train_loader, val_loader = create_sample_data()
    
    # Create initial network
    print("\n🏗️ Creating initial sparse network...")
    network = create_standard_network(
        architecture=[784, 128, 64, 10],
        sparsity=0.1,
        device=str(device)
    )
    
    print(f"Initial network: {[784, 128, 64, 10]} with 10% sparsity")
    
    # Create composable evolution system
    evolution_system = create_standard_evolution_system()
    
    # Create autocorrelation framework
    performance_analyzer = PerformanceAnalyzer()
    
    # Create metrics system
    threshold_config = ThresholdConfig(adaptive=True, activation_threshold=0.01)
    metrics_config = MetricsConfig(compute_mi=True, compute_activity=True, compute_sensli=True, compute_graph=True)
    metrics_system = CompleteMetricsSystem(network, threshold_config, metrics_config)
    
    print("\n🧠 Initializing Composable Evolution System with Autocorrelation Framework...")
    
    # Run growth with autocorrelation learning
    print("\n🌱 Starting growth with meta-learning...")
    
    try:
        context = NetworkContext(network, train_loader, device, {'val_loader': val_loader})
        for i in range(3): # Small number for demo
            context = evolution_system.evolve_network(context, num_iterations=1)
            
            # Collect data for autocorrelation
            performance_metrics = {
                'train_acc': context.performance_history[-1] if context.performance_history else 0.0,
                'val_acc': context.performance_history[-1] if context.performance_history else 0.0,
            }
            performance_analyzer.collect_checkpoint_data(
                network=context.network,
                dataloader=train_loader,
                epoch=i,
                performance_metrics=performance_metrics
            )
            complete_metrics = metrics_system.compute_all_metrics(train_loader)
            performance_analyzer.update_metrics_from_complete_system(i, complete_metrics)

        print("\n🎉 Growth complete!")
        
        # Demonstrate learned insights
        print("\n🔍 Analyzing learned patterns...")
        
        # Get strategy effectiveness summary
        effectiveness = performance_analyzer.get_strategy_effectiveness_summary()
        if effectiveness:
            print("\n📈 Strategy Effectiveness Learned:")
            for strategy, stats in effectiveness.items():
                print(f"  {strategy}:")
                print(f"    Success Rate: {stats['success_rate']:.1%}")
                print(f"    Avg Improvement: {stats['avg_improvement']:+.3f}")
        
        # Show correlation insights if available
        correlation_results = performance_analyzer.analyze_metric_correlations()
        if correlation_results:
            print("\n🧬 Top Predictive Metrics Discovered:")
            top_metrics = performance_analyzer._find_top_predictive_metrics(
                correlation_results, top_n=3
            )
            for i, metric_info in enumerate(top_metrics):
                print(f"  {i+1}. {metric_info['metric']}")
                print(f"     Correlation: {metric_info['val_correlation']:.3f}")
                print(f"     Significant: {metric_info['significant']}")
        
    except Exception as e:
        print(f"\n❌ Error during growth: {e}")
        print("This might be due to limited data or computational constraints in the demo.")
        print("The framework is designed to work with larger datasets and longer training.")

if __name__ == "__main__":
    main()

