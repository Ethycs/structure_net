#!/usr/bin/env python3
"""
Example demonstrating the new modular metrics system and autocorrelation framework.

This example shows how to use the refactored metrics system with improved
performance and the new autocorrelation framework for meta-learning.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Import the new modular components
from structure_net.evolution.metrics import (
    CompleteMetricsSystem, 
    ThresholdConfig, 
    MetricsConfig,
    MutualInformationAnalyzer,
    ActivityAnalyzer
)

# Import autocorrelation framework
try:
    from structure_net.evolution.autocorrelation import PerformanceAnalyzer
    AUTOCORR_AVAILABLE = True
except ImportError:
    AUTOCORR_AVAILABLE = False
    print("⚠️  Autocorrelation framework not available")

from structure_net.core.network_factory import create_standard_network


def create_sample_data(batch_size=64, input_size=784, num_classes=10, num_samples=1000):
    """Create sample data for demonstration."""
    X = torch.randn(num_samples, input_size)
    y = torch.randint(0, num_classes, (num_samples,))
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def demonstrate_modular_metrics():
    """Demonstrate the new modular metrics system."""
    print("🔬 Demonstrating Modular Metrics System")
    print("=" * 50)
    
    # Create a sample network
    network = create_standard_network(784, [128, 64], 10, sparsity=0.1)
    
    # Configure the metrics system
    threshold_config = ThresholdConfig(
        activation_threshold=0.01,
        weight_threshold=0.001,
        persistence_ratio=0.1,
        adaptive=True
    )
    
    metrics_config = MetricsConfig(
        compute_mi=True,
        compute_activity=True,
        compute_sensli=True,
        compute_graph=True,
        max_batches=5,
        sensli_optimization=True
    )
    
    # Create the integrated metrics system
    metrics_system = CompleteMetricsSystem(network, threshold_config, metrics_config)
    
    # Create sample data
    data_loader = create_sample_data()
    
    print("📊 Computing comprehensive metrics...")
    
    # Compute all metrics
    results = metrics_system.compute_all_metrics(data_loader, num_batches=3)
    
    print("\n✅ Metrics computation complete!")
    print(f"📈 Results summary:")
    print(f"  - MI metrics computed for {len(results['mi_metrics'])} layer pairs")
    print(f"  - Activity metrics computed for {len(results['activity_metrics'])} layers")
    print(f"  - SensLI metrics computed for {len(results['sensli_metrics'])} layer pairs")
    print(f"  - Graph metrics: {results['graph_metrics'].get('num_nodes', 0)} nodes, {results['graph_metrics'].get('num_edges', 0)} edges")
    
    # Show summary metrics
    summary = results['summary']
    if summary:
        print(f"\n📋 Summary:")
        print(f"  - Average MI efficiency: {summary.get('avg_mi_efficiency', 0):.3f}")
        print(f"  - Average active ratio: {summary.get('avg_active_ratio', 0):.3f}")
        print(f"  - Critical bottlenecks: {summary.get('critical_bottlenecks', 0)}")
        print(f"  - Network connected: {summary.get('network_connected', False)}")
    
    # Show computation statistics
    comp_stats = results.get('computation_stats', {})
    if comp_stats:
        print(f"\n⚡ Performance Statistics:")
        for analyzer, stats in comp_stats.items():
            if stats:
                print(f"  - {analyzer}: {stats.get('total_calls', 0)} calls, "
                      f"{stats.get('avg_time_per_call', 0):.4f}s avg, "
                      f"{stats.get('cache_hit_rate', 0):.1%} cache hit rate")
    
    return metrics_system, results


def demonstrate_individual_analyzers():
    """Demonstrate using individual analyzers directly."""
    print("\n🔧 Demonstrating Individual Analyzers")
    print("=" * 50)
    
    # Create sample data
    X = torch.randn(100, 50)  # 100 samples, 50 features
    Y = torch.randn(100, 30)  # 100 samples, 30 features
    
    # 1. Mutual Information Analyzer
    print("📊 Testing Mutual Information Analyzer...")
    threshold_config = ThresholdConfig(activation_threshold=0.01)
    mi_analyzer = MutualInformationAnalyzer(threshold_config)
    
    mi_results = mi_analyzer.compute_metrics(X, Y)
    print(f"  - MI: {mi_results['mi']:.4f}")
    print(f"  - MI Efficiency: {mi_results['mi_efficiency']:.4f}")
    print(f"  - Method: {mi_results['method']}")
    
    # 2. Activity Analyzer
    print("\n🎯 Testing Activity Analyzer...")
    activity_analyzer = ActivityAnalyzer(threshold_config)
    
    activations = torch.randn(64, 128) * 2  # Some activations
    activity_results = activity_analyzer.compute_metrics(activations, layer_idx=0)
    print(f"  - Active ratio: {activity_results['active_ratio']:.3f}")
    print(f"  - Layer health score: {activity_results['layer_health_score']:.3f}")
    print(f"  - Activity entropy: {activity_results['activity_entropy']:.3f}")


def demonstrate_autocorrelation_framework():
    """Demonstrate the autocorrelation framework for meta-learning."""
    if not AUTOCORR_AVAILABLE:
        print("\n⚠️  Autocorrelation framework not available - skipping demonstration")
        return
    
    print("\n🧠 Demonstrating Autocorrelation Framework")
    print("=" * 50)
    
    # Create performance analyzer
    performance_analyzer = PerformanceAnalyzer()
    
    # Simulate collecting data over multiple epochs
    print("📊 Simulating training data collection...")
    
    for epoch in range(25):
        # Simulate performance metrics
        performance_metrics = {
            'train_acc': 0.5 + 0.4 * (epoch / 25) + 0.05 * torch.randn(1).item(),
            'val_acc': 0.45 + 0.35 * (epoch / 25) + 0.05 * torch.randn(1).item(),
            'train_loss': 2.0 - 1.5 * (epoch / 25) + 0.1 * torch.randn(1).item(),
            'val_loss': 2.1 - 1.4 * (epoch / 25) + 0.1 * torch.randn(1).item()
        }
        
        # Collect checkpoint data
        performance_analyzer.collect_checkpoint_data(
            network=None,  # Would be actual network
            dataloader=None,  # Would be actual dataloader
            epoch=epoch,
            performance_metrics=performance_metrics
        )
        
        # Simulate updating with complete metrics
        complete_metrics = {
            'mi_metrics': {
                'layer_0_1': {'mi_efficiency': 0.3 + 0.2 * torch.randn(1).item()}
            },
            'activity_metrics': {
                'layer_0': {'active_ratio': 0.8 + 0.1 * torch.randn(1).item()}
            },
            'graph_metrics': {
                'algebraic_connectivity': 0.1 + 0.05 * torch.randn(1).item()
            }
        }
        
        performance_analyzer.update_metrics_from_complete_system(epoch, complete_metrics)
    
    print(f"✅ Collected data for {len(performance_analyzer.metric_history)} epochs")
    
    # Analyze correlations
    print("\n🔍 Analyzing metric-performance correlations...")
    correlation_results = performance_analyzer.analyze_metric_correlations(min_history_length=15)
    
    if correlation_results:
        top_metrics = correlation_results.get('top_predictive_metrics', [])
        print(f"📈 Found {len(top_metrics)} predictive metrics:")
        
        for i, metric_info in enumerate(top_metrics[:5]):
            print(f"  {i+1}. {metric_info['metric']}: correlation = {metric_info.get('val_correlation', 0):.3f}")
        
        # Get growth recommendations
        current_metrics = {
            'mi_efficiency_mean': 0.25,
            'active_neuron_ratio': 0.75,
            'algebraic_connectivity': 0.08
        }
        
        recommendations = performance_analyzer.get_growth_recommendations(current_metrics)
        print(f"\n💡 Growth recommendations:")
        for i, rec in enumerate(recommendations):
            print(f"  {i+1}. {rec['action']} (confidence: {rec['confidence']:.2f})")
            print(f"     Reason: {rec['reason']}")
    
    # Show insights summary
    insights = performance_analyzer.get_insights_summary()
    print(f"\n📋 Insights Summary:")
    print(f"  - Data points collected: {insights['data_collection']['total_checkpoints']}")
    print(f"  - Metrics analyzed: {insights['correlation_analysis']['total_metrics_analyzed']}")
    print(f"  - Significant correlations: {insights['correlation_analysis']['significant_correlations']}")


def main():
    """Run all demonstrations."""
    print("🚀 Modular Metrics System Demonstration")
    print("=" * 60)
    
    # Demonstrate the integrated system
    metrics_system, results = demonstrate_modular_metrics()
    
    # Demonstrate individual analyzers
    demonstrate_individual_analyzers()
    
    # Demonstrate autocorrelation framework
    demonstrate_autocorrelation_framework()
    
    print("\n" + "=" * 60)
    print("✅ All demonstrations completed successfully!")
    print("\nKey benefits of the modular system:")
    print("  🔧 Modular design for better maintainability")
    print("  ⚡ Optimized data collection (single pass)")
    print("  🧠 Autocorrelation framework for meta-learning")
    print("  📊 Enhanced caching and performance statistics")
    print("  🔄 Full backward compatibility")
    print("  🎯 Specialized analyzers for focused analysis")


if __name__ == "__main__":
    main()
