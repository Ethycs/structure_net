#!/usr/bin/env python3
"""
Example: Using the Autocorrelation Framework for Meta-Learning Growth

This example demonstrates how to use the new MetricPerformanceAnalyzer
to discover which metrics predict learning success and automatically
optimize growth strategies based on learned patterns.
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from structure_net.core.network_factory import create_standard_network
from structure_net.evolution.integrated_growth_system import IntegratedGrowthSystem
from structure_net.evolution.advanced_layers import ThresholdConfig, MetricsConfig

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
        input_size=784,
        hidden_sizes=[128, 64],
        output_size=10,
        sparsity=0.1
    ).to(device)
    
    print(f"Initial network: {[784, 128, 64, 10]} with 10% sparsity")
    
    # Configure thresholds and metrics
    threshold_config = ThresholdConfig()
    threshold_config.adaptive = True  # Enable adaptive thresholds
    threshold_config.activation_threshold = 0.01
    
    metrics_config = MetricsConfig()
    metrics_config.compute_mi = True
    metrics_config.compute_activity = True
    metrics_config.compute_sensli = True
    metrics_config.compute_graph = True
    
    # Create integrated growth system with autocorrelation framework
    print("\n🧠 Initializing Integrated Growth System with Autocorrelation Framework...")
    growth_system = IntegratedGrowthSystem(
        network=network,
        config=threshold_config,
        metrics_config=metrics_config
    )
    
    print("✅ System initialized with:")
    print("   - MetricPerformanceAnalyzer for correlation discovery")
    print("   - Learned strategy weighting")
    print("   - Comprehensive metrics collection")
    print("   - Adaptive threshold management")
    
    # Run growth with autocorrelation learning
    print("\n🌱 Starting growth with meta-learning...")
    print("This will:")
    print("  1. Collect comprehensive metrics at each step")
    print("  2. Analyze correlations between metrics and performance")
    print("  3. Learn which strategies work best under which conditions")
    print("  4. Automatically weight strategies based on learned patterns")
    print("  5. Provide insights into the 'laws of neural network growth'")
    
    try:
        final_network = growth_system.grow_network(
            train_loader=train_loader,
            val_loader=val_loader,
            growth_iterations=3,  # Small number for demo
            epochs_per_iteration=10,  # Reduced for faster execution
            tournament_epochs=3
        )
        
        print("\n🎉 Growth complete!")
        
        # Demonstrate learned insights
        print("\n🔍 Analyzing learned patterns...")
        
        # Get strategy effectiveness summary
        effectiveness = growth_system.performance_analyzer.get_strategy_effectiveness_summary()
        if effectiveness:
            print("\n📈 Strategy Effectiveness Learned:")
            for strategy, stats in effectiveness.items():
                print(f"  {strategy}:")
                print(f"    Success Rate: {stats['success_rate']:.1%}")
                print(f"    Avg Improvement: {stats['avg_improvement']:+.3f}")
        
        # Show correlation insights if available
        if growth_system.performance_analyzer.correlation_results:
            print("\n🧬 Top Predictive Metrics Discovered:")
            top_metrics = growth_system.performance_analyzer._find_top_predictive_metrics(
                growth_system.performance_analyzer.correlation_results, top_n=3
            )
            for i, metric_info in enumerate(top_metrics):
                print(f"  {i+1}. {metric_info['metric']}")
                print(f"     Correlation: {metric_info['val_correlation']:.3f}")
                print(f"     Significant: {metric_info['significant']}")
        
        # Show learned strategy weights
        print("\n⚖️ Learned Strategy Weights:")
        for strategy, weight in growth_system.learned_strategy_weights.items():
            print(f"  {strategy}: {weight:.2f}")
        
        print("\n✨ Key Insights:")
        print("  - The system learned which metrics predict performance improvements")
        print("  - Strategy weights were automatically adjusted based on effectiveness")
        print("  - Future growth decisions will be guided by these learned patterns")
        print("  - This creates a self-improving growth system!")
        
    except Exception as e:
        print(f"\n❌ Error during growth: {e}")
        print("This might be due to limited data or computational constraints in the demo.")
        print("The framework is designed to work with larger datasets and longer training.")

if __name__ == "__main__":
    main()
