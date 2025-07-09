#!/usr/bin/env python3
"""
Comprehensive Example: Advanced Network Growth with All Metrics

This example demonstrates the full capabilities of the integrated growth system,
including detailed analysis, all metrics, and step-by-step growth process.
"""

from src.structure_net import (
    create_standard_network, 
    analyze_and_grow_network, 
    StructureNetGrowthSystem,
    ThresholdConfig, 
    MetricsConfig,
    ExactMutualInformation,
    get_network_stats
)
import torch
import torch.nn.functional as F
import numpy as np

def create_realistic_data_loader(dataset='mnist', batch_size=32, num_samples=1000):
    """Create a more realistic data loader for testing."""
    if dataset == 'mnist':
        # MNIST-like data
        data = torch.randn(num_samples, 784) * 0.5 + 0.1
        # Add some structure to make it more realistic
        data[:, :100] += torch.randn(num_samples, 100) * 0.3  # Some correlated features
        data = torch.clamp(data, 0, 1)  # Normalize like MNIST
        labels = torch.randint(0, 10, (num_samples,))
    else:
        # CIFAR-10-like data
        data = torch.randn(num_samples, 3072) * 0.5 + 0.1
        data = torch.clamp(data, 0, 1)
        labels = torch.randint(0, 10, (num_samples,))
    
    dataset = torch.utils.data.TensorDataset(data, labels)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

def print_detailed_network_stats(network, name="Network"):
    """Print comprehensive network statistics."""
    stats = get_network_stats(network)
    
    print(f"\n📊 {name} Statistics:")
    print("=" * 50)
    print(f"   Architecture: {stats['architecture']}")
    print(f"   Total parameters: {stats['total_parameters']:,}")
    print(f"   Total connections: {stats['total_connections']:,}")
    print(f"   Overall sparsity: {stats['overall_sparsity']:.1%}")
    
    print(f"\n   Layer Details:")
    for i, layer_stats in enumerate(stats['layers']):
        print(f"     Layer {i}: {layer_stats['in_features']}→{layer_stats['out_features']}")
        print(f"       Active connections: {layer_stats['active_connections']:,}")
        print(f"       Sparsity: {layer_stats['sparsity_ratio']:.1%}")
        print(f"       Parameters: {layer_stats['parameters']:,}")

def demonstrate_exact_mi_analysis(network, data_loader):
    """Demonstrate exact mutual information analysis."""
    print(f"\n🔬 EXACT MUTUAL INFORMATION ANALYSIS")
    print("=" * 60)
    
    # Create MI analyzer
    mi_analyzer = ExactMutualInformation(threshold=0.01)
    
    # Get activations from network
    network.eval()
    activations = []
    
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(data_loader):
            if batch_idx >= 5:  # Use 5 batches
                break
            
            x = data.view(data.size(0), -1)
            layer_acts = []
            
            for layer in network:
                if hasattr(layer, 'linear'):  # StandardSparseLayer
                    x = layer(x)
                    layer_acts.append(x.clone())
                    x = F.relu(x)
                elif isinstance(layer, torch.nn.ReLU):
                    x = layer(x)
            
            activations.append(layer_acts)
    
    # Analyze MI between consecutive layers
    print(f"\n📈 Layer-by-Layer MI Analysis:")
    for layer_idx in range(len(activations[0]) - 1):
        # Concatenate activations across batches
        acts_i = torch.cat([batch[layer_idx] for batch in activations], dim=0)
        acts_j = torch.cat([batch[layer_idx + 1] for batch in activations], dim=0)
        
        # Apply ReLU to intermediate layers
        if layer_idx < len(activations[0]) - 2:
            acts_i = F.relu(acts_i)
        if layer_idx + 1 < len(activations[0]) - 1:
            acts_j = F.relu(acts_j)
        
        # Compute MI
        mi_result = mi_analyzer.compute_exact_mi(acts_i, acts_j)
        
        print(f"\n   Layer {layer_idx} → Layer {layer_idx + 1}:")
        print(f"     MI: {mi_result['mi']:.4f} bits")
        print(f"     Normalized MI: {mi_result['normalized_mi']:.4f}")
        print(f"     Method: {mi_result['method']}")
        print(f"     Active neurons: {mi_result['active_neurons_X']} → {mi_result['active_neurons_Y']}")
        print(f"     Sparsity: {mi_result['sparsity_X']:.1%} → {mi_result['sparsity_Y']:.1%}")
        print(f"     Entropy X: {mi_result['entropy_X']:.4f} bits")
        print(f"     Entropy Y: {mi_result['entropy_Y']:.4f} bits")
        
        # Calculate efficiency
        max_entropy = min(mi_result['entropy_X'], mi_result['entropy_Y'])
        efficiency = mi_result['mi'] / (max_entropy + 1e-10)
        print(f"     MI Efficiency: {efficiency:.2%}")
        
        if efficiency < 0.3:
            print(f"     ⚠️  BOTTLENECK DETECTED!")
        elif efficiency > 0.7:
            print(f"     ✅ Good information flow")

def demonstrate_comprehensive_analysis(network, data_loader):
    """Demonstrate the full growth system with all metrics."""
    print(f"\n🧬 COMPREHENSIVE GROWTH SYSTEM ANALYSIS")
    print("=" * 60)
    
    # Configure thresholds and metrics
    threshold_config = ThresholdConfig()
    threshold_config.activation_threshold = 0.01
    threshold_config.gradient_threshold = 0.001
    threshold_config.adaptive = True
    
    metrics_config = MetricsConfig()
    metrics_config.compute_betweenness = True
    metrics_config.compute_spectral = True
    metrics_config.compute_paths = True
    
    # Create growth system
    growth_system = StructureNetGrowthSystem(
        network, 
        threshold_config, 
        metrics_config
    )
    
    # Perform comprehensive analysis
    print(f"\n🔍 Running comprehensive bottleneck analysis...")
    analysis_results = growth_system.analyze_network_bottlenecks(
        data_loader, 
        num_batches=10
    )
    
    # Print detailed results
    print(f"\n📊 DETAILED ANALYSIS RESULTS")
    print("=" * 50)
    
    # Layer-by-layer analysis
    print(f"\n🔬 Layer-by-Layer Analysis:")
    for layer_pair, analysis in analysis_results['layer_analyses'].items():
        if 'error' in analysis:
            continue
            
        print(f"\n   {layer_pair}:")
        print(f"     MI: {analysis['mi']:.4f} bits")
        print(f"     Normalized MI: {analysis['normalized_mi']:.4f}")
        print(f"     MI Efficiency: {analysis['mi_efficiency']:.2%}")
        print(f"     Method: {analysis['method']}")
        print(f"     Active neurons: {analysis['active_neurons_input']} → {analysis['active_neurons_output']}")
        print(f"     Dead ratios: {analysis['dead_ratio_input']:.1%} → {analysis['dead_ratio_output']:.1%}")
        print(f"     Entropies: {analysis['entropy_input']:.3f} → {analysis['entropy_output']:.3f} bits")
    
    # Bottleneck summary
    print(f"\n🚨 Bottleneck Summary:")
    if analysis_results['bottlenecks']:
        for i, bottleneck in enumerate(analysis_results['bottlenecks']):
            print(f"   {i+1}. Layer {bottleneck['layer_pair'][0]}→{bottleneck['layer_pair'][1]}")
            print(f"      Type: {bottleneck['type']}")
            print(f"      Severity: {bottleneck['severity']:.2f}")
            print(f"      MI Efficiency: {bottleneck['mi_efficiency']:.2%}")
    else:
        print("   ✅ No significant bottlenecks detected")
    
    # Dead zone summary
    print(f"\n💀 Dead Zone Summary:")
    if analysis_results['dead_zones']:
        for i, dead_zone in enumerate(analysis_results['dead_zones']):
            print(f"   {i+1}. Layer {dead_zone['layer_pair'][0]}→{dead_zone['layer_pair'][1]}")
            print(f"      Input dead ratio: {dead_zone['dead_ratio_input']:.1%}")
            print(f"      Output dead ratio: {dead_zone['dead_ratio_output']:.1%}")
    else:
        print("   ✅ No significant dead zones detected")
    
    # Recommendations
    print(f"\n💡 Growth Recommendations:")
    if analysis_results['recommendations']:
        for i, rec in enumerate(analysis_results['recommendations']):
            print(f"   {i+1}. {rec['action']} (Priority: {rec['priority']})")
            print(f"      Reason: {rec['reason']}")
            print(f"      Expected improvement: {rec['expected_improvement']:.2f}")
            if 'position' in rec:
                print(f"      Position: {rec['position']}")
            if 'factor' in rec:
                print(f"      Factor: {rec['factor']}")
    else:
        print("   ✅ No growth recommendations needed")
    
    return analysis_results, growth_system

def demonstrate_step_by_step_growth(network, data_loader):
    """Demonstrate step-by-step network growth with detailed metrics."""
    print(f"\n🌱 STEP-BY-STEP NETWORK GROWTH")
    print("=" * 60)
    
    current_network = network
    
    for iteration in range(3):
        print(f"\n🔄 Growth Iteration {iteration + 1}")
        print("-" * 40)
        
        # Print current network stats
        print_detailed_network_stats(current_network, f"Iteration {iteration + 1} Network")
        
        # Analyze and grow
        print(f"\n🔍 Analyzing network...")
        analysis_results, growth_system = demonstrate_comprehensive_analysis(current_network, data_loader)
        
        if analysis_results['recommendations']:
            print(f"\n🌱 Applying growth recommendations...")
            improved_network = growth_system.apply_growth_recommendations(
                analysis_results['recommendations'], 
                max_actions=2
            )
            
            # Check if network actually changed
            old_stats = get_network_stats(current_network)
            new_stats = get_network_stats(improved_network)
            
            if old_stats['architecture'] != new_stats['architecture']:
                print(f"\n📈 Network Growth Applied:")
                print(f"   Before: {old_stats['architecture']}")
                print(f"   After:  {new_stats['architecture']}")
                print(f"   Parameter change: {new_stats['total_parameters'] - old_stats['total_parameters']:+,}")
                current_network = improved_network
            else:
                print(f"\n📊 No architectural changes applied")
                break
        else:
            print(f"\n✅ Network is already optimal - no growth needed")
            break
    
    return current_network

def main():
    """Main demonstration function."""
    print("🚀 COMPREHENSIVE INTEGRATED GROWTH SYSTEM DEMONSTRATION")
    print("=" * 80)
    
    # Create initial network
    print(f"\n🏗️  Creating initial network...")
    network = create_standard_network([784, 128, 64, 10], sparsity=0.02, device='cpu')
    print_detailed_network_stats(network, "Initial Network")
    
    # Create realistic data
    print(f"\n📦 Creating realistic data loader...")
    train_loader = create_realistic_data_loader('mnist', batch_size=32, num_samples=500)
    print(f"   ✅ Created MNIST-like dataset with 500 samples")
    
    # Demonstrate exact MI analysis
    demonstrate_exact_mi_analysis(network, train_loader)
    
    # Demonstrate comprehensive analysis
    print(f"\n" + "="*80)
    analysis_results, growth_system = demonstrate_comprehensive_analysis(network, train_loader)
    
    # Demonstrate step-by-step growth
    print(f"\n" + "="*80)
    final_network = demonstrate_step_by_step_growth(network, train_loader)
    
    # Final comparison
    print(f"\n🏁 FINAL COMPARISON")
    print("=" * 50)
    print_detailed_network_stats(network, "Original Network")
    print_detailed_network_stats(final_network, "Final Grown Network")
    
    # Calculate improvement metrics
    original_stats = get_network_stats(network)
    final_stats = get_network_stats(final_network)
    
    param_increase = final_stats['total_parameters'] - original_stats['total_parameters']
    connection_increase = final_stats['total_connections'] - original_stats['total_connections']
    
    print(f"\n📈 Growth Summary:")
    print(f"   Architecture: {original_stats['architecture']} → {final_stats['architecture']}")
    print(f"   Parameters: {original_stats['total_parameters']:,} → {final_stats['total_parameters']:,} ({param_increase:+,})")
    print(f"   Connections: {original_stats['total_connections']:,} → {final_stats['total_connections']:,} ({connection_increase:+,})")
    print(f"   Layers added: {len(final_stats['architecture']) - len(original_stats['architecture'])}")
    
    print(f"\n🎯 Demonstration complete!")
    print(f"   The integrated growth system successfully analyzed and improved the network")
    print(f"   using exact mutual information and information theory principles.")

if __name__ == "__main__":
    main()
