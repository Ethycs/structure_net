#!/usr/bin/env python3
"""
Composable Evolution System Example

This example demonstrates the new interface-based composable evolution system.
It shows how to:
1. Create different evolution systems by composing components
2. Configure components individually
3. Monitor component performance
4. Compare different evolutionary approaches

The composable system eliminates hardcoded strategies and enables
flexible experimentation with different combinations of analyzers,
growth strategies, and training approaches.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Import the new composable system
from src.structure_net.evolution.components import (
    # Core interfaces
    NetworkContext, 
    
    # Analyzers
    StandardExtremaAnalyzer,
    NetworkStatsAnalyzer,
    SimpleInformationFlowAnalyzer,
    
    # Strategies
    ExtremaGrowthStrategy,
    InformationFlowGrowthStrategy,
    ResidualBlockGrowthStrategy,
    HybridGrowthStrategy,
    
    # Evolution systems
    ComposableEvolutionSystem,
    create_standard_evolution_system,
    create_extrema_focused_system,
    create_hybrid_system
)

# Import existing infrastructure
from src.structure_net.core.network_factory import create_standard_network


def create_sample_dataset(num_samples: int = 1000, input_dim: int = 784, num_classes: int = 10):
    """Create a simple synthetic dataset for demonstration."""
    # Generate random data
    X = torch.randn(num_samples, input_dim)
    
    # Create simple patterns for classification
    # Class 0: positive values in first quarter
    # Class 1: positive values in second quarter, etc.
    y = torch.zeros(num_samples, dtype=torch.long)
    
    quarter_size = input_dim // 4
    for i in range(min(4, num_classes)):
        start_idx = i * quarter_size
        end_idx = (i + 1) * quarter_size
        
        # Samples where this quarter has mostly positive values
        quarter_positive = (X[:, start_idx:end_idx] > 0).float().mean(dim=1) > 0.6
        y[quarter_positive] = i
    
    # Random assignment for remaining classes
    for i in range(4, num_classes):
        mask = (torch.rand(num_samples) < 0.1) & (y == 0)
        y[mask] = i
    
    return TensorDataset(X, y)


def demonstrate_basic_composable_system():
    """Demonstrate basic usage of the composable evolution system."""
    print("\n" + "="*80)
    print("🧬 BASIC COMPOSABLE EVOLUTION SYSTEM DEMO")
    print("="*80)
    
    # Create dataset
    dataset = create_sample_dataset(num_samples=500, input_dim=784, num_classes=10)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Create initial network
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    network = create_standard_network(
        architecture=[784, 128, 64, 10],
        sparsity=0.1,
        device=str(device)
    )
    
    print(f"📊 Initial network: [784, 128, 64, 10], sparsity=10%, device={device}")
    
    # Create network context
    context = NetworkContext(
        network=network,
        data_loader=data_loader,
        device=device
    )
    
    # Create composable evolution system manually
    print("\n🔧 Building composable system...")
    system = ComposableEvolutionSystem()
    
    # Add analyzers
    system.add_component(StandardExtremaAnalyzer(max_batches=3))
    system.add_component(NetworkStatsAnalyzer())
    system.add_component(SimpleInformationFlowAnalyzer())
    
    # Add growth strategies
    system.add_component(ExtremaGrowthStrategy(extrema_threshold=0.25))
    system.add_component(InformationFlowGrowthStrategy())
    
    print(f"   Added {len(system.analyzers)} analyzers")
    print(f"   Added {len(system.strategies)} growth strategies")
    print(f"   Added {len(system.trainers)} trainers (auto-added)")
    
    # Run evolution
    print("\n🚀 Starting evolution...")
    evolved_context = system.evolve_network(context, num_iterations=3)
    
    # Show results
    print("\n📈 Evolution Results:")
    summary = system.get_evolution_summary()
    print(f"   Total iterations: {summary['total_iterations']}")
    print(f"   Growth events: {summary['metrics'].get('total_growth_events', 0)}")
    print(f"   Average iteration time: {summary['metrics'].get('average_iteration_time', 0):.1f}s")
    
    # Show component metrics
    print("\n📊 Component Metrics:")
    all_metrics = system.get_metrics()
    for key, value in all_metrics.items():
        if isinstance(value, (int, float)):
            print(f"   {key}: {value}")
    
    return system, evolved_context


def demonstrate_preconfigured_systems():
    """Demonstrate the preconfigured evolution systems."""
    print("\n" + "="*80)
    print("🏭 PRECONFIGURED EVOLUTION SYSTEMS DEMO")
    print("="*80)
    
    # Create dataset
    dataset = create_sample_dataset(num_samples=300, input_dim=784, num_classes=10)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    systems = {
        "Standard System": create_standard_evolution_system(),
        "Extrema-Focused": create_extrema_focused_system(),
        "Hybrid System": create_hybrid_system()
    }
    
    results = {}
    
    for system_name, system in systems.items():
        print(f"\n🔬 Testing {system_name}...")
        
        # Create fresh network for each system
        network = create_standard_network(
            architecture=[784, 64, 32, 10],
            sparsity=0.15,
            device=str(device)
        )
        
        context = NetworkContext(
            network=network,
            data_loader=data_loader,
            device=device
        )
        
        # Run evolution
        evolved_context = system.evolve_network(context, num_iterations=2)
        
        # Collect results
        summary = system.get_evolution_summary()
        results[system_name] = {
            'growth_events': summary['metrics'].get('total_growth_events', 0),
            'final_performance': evolved_context.performance_history[-1] if evolved_context.performance_history else 0.0,
            'components': summary['components']
        }
        
        print(f"   Growth events: {results[system_name]['growth_events']}")
        print(f"   Final accuracy: {results[system_name]['final_performance']:.2%}")
    
    # Compare results
    print("\n📊 System Comparison:")
    print("-" * 60)
    print(f"{'System':<20} {'Growth Events':<15} {'Final Acc':<12} {'Components'}")
    print("-" * 60)
    
    for system_name, result in results.items():
        components = result['components']
        comp_str = f"{components['analyzers']}A/{components['strategies']}S"
        print(f"{system_name:<20} {result['growth_events']:<15} "
              f"{result['final_performance']:<12.2%} {comp_str}")
    
    return results


def demonstrate_component_configuration():
    """Demonstrate component configuration and customization."""
    print("\n" + "="*80)
    print("⚙️  COMPONENT CONFIGURATION DEMO")
    print("="*80)
    
    # Create system
    system = ComposableEvolutionSystem()
    
    # Add components with custom configuration
    print("🔧 Adding and configuring components...")
    
    # Extrema analyzer with custom settings
    extrema_analyzer = StandardExtremaAnalyzer()
    extrema_analyzer.configure({
        'dead_threshold': 0.005,  # More sensitive
        'saturated_multiplier': 3.0,  # Higher threshold
        'max_batches': 8  # More data
    })
    system.add_component(extrema_analyzer)
    
    # Growth strategy with custom settings
    extrema_strategy = ExtremaGrowthStrategy()
    extrema_strategy.configure({
        'extrema_threshold': 0.2,  # Lower threshold
        'dead_neuron_threshold': 3,  # Fewer neurons needed
        'patch_size': 5  # Larger patches
    })
    system.add_component(extrema_strategy)
    
    # Add other components
    system.add_component(NetworkStatsAnalyzer())
    
    print(f"   Configured {len(system.get_components())} components")
    
    # Show configurations
    print("\n📋 Component Configurations:")
    config = system.get_configuration()
    for component_type, configs in config.items():
        if configs:
            print(f"   {component_type}:")
            for comp_name, comp_config in configs.items():
                print(f"     {comp_name}: {comp_config}")
    
    # Test with data
    dataset = create_sample_dataset(num_samples=200)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    network = create_standard_network([784, 96, 48, 10], sparsity=0.2, device=str(device))
    context = NetworkContext(network=network, data_loader=data_loader, device=device)
    
    print("\n🧪 Testing configured system...")
    evolved_context = system.evolve_network(context, num_iterations=2)
    
    # Show component-specific metrics
    print("\n📊 Component Performance:")
    metrics = system.get_metrics()
    for key, value in metrics.items():
        if 'analyzer' in key or 'strategy' in key:
            print(f"   {key}: {value}")
    
    return system


def demonstrate_custom_hybrid_strategy():
    """Demonstrate creating custom hybrid strategies."""
    print("\n" + "="*80)
    print("🔀 CUSTOM HYBRID STRATEGY DEMO")
    print("="*80)
    
    # Create individual strategies with different configurations
    print("🏗️  Building custom hybrid strategy...")
    
    # Aggressive extrema strategy
    aggressive_extrema = ExtremaGrowthStrategy()
    aggressive_extrema.configure({
        'extrema_threshold': 0.15,  # Very sensitive
        'dead_neuron_threshold': 2,
        'patch_size': 4
    })
    
    # Conservative information flow strategy
    conservative_info = InformationFlowGrowthStrategy()
    conservative_info.configure({
        'bottleneck_threshold': 0.2,  # Less sensitive
        'efficiency_threshold': 0.6
    })
    
    # Residual block strategy for deep networks
    residual_strategy = ResidualBlockGrowthStrategy()
    residual_strategy.configure({
        'num_layers': 3,  # Larger blocks
        'activation_threshold': 0.15
    })
    
    # Create hybrid strategy
    hybrid_strategy = HybridGrowthStrategy([
        aggressive_extrema,
        conservative_info,
        residual_strategy
    ])
    
    print(f"   Combined {len(hybrid_strategy.strategies)} strategies")
    
    # Create system with hybrid strategy
    system = ComposableEvolutionSystem()
    system.add_component(StandardExtremaAnalyzer())
    system.add_component(NetworkStatsAnalyzer())
    system.add_component(SimpleInformationFlowAnalyzer())
    system.add_component(hybrid_strategy)
    
    # Test the hybrid system
    dataset = create_sample_dataset(num_samples=400)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    network = create_standard_network([784, 128, 64, 32, 10], sparsity=0.1, device=str(device))
    context = NetworkContext(network=network, data_loader=data_loader, device=device)
    
    print("\n🚀 Testing hybrid strategy...")
    evolved_context = system.evolve_network(context, num_iterations=3)
    
    # Show hybrid strategy metrics
    print("\n📊 Hybrid Strategy Performance:")
    hybrid_metrics = hybrid_strategy.get_metrics()
    for key, value in hybrid_metrics.items():
        print(f"   {key}: {value}")
    
    return system, hybrid_strategy


def main():
    """Run all demonstrations."""
    print("🧬 COMPOSABLE EVOLUTION SYSTEM DEMONSTRATIONS")
    print("=" * 80)
    print("This example shows the new interface-based composable evolution system.")
    print("Key benefits:")
    print("• Modular components that can be mixed and matched")
    print("• Individual component configuration")
    print("• Monitoring and metrics for each component")
    print("• Easy experimentation with different approaches")
    print("• No hardcoded strategies - everything is composable!")
    
    try:
        # Run demonstrations
        demonstrate_basic_composable_system()
        demonstrate_preconfigured_systems()
        demonstrate_component_configuration()
        demonstrate_custom_hybrid_strategy()
        
        print("\n" + "="*80)
        print("✅ ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\nKey takeaways:")
        print("• The composable system eliminates hardcoded strategies")
        print("• Components can be configured individually for fine-tuning")
        print("• Different evolution approaches can be easily compared")
        print("• Hybrid strategies combine multiple approaches intelligently")
        print("• The system is fully modular and extensible")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
