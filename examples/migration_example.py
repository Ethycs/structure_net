#!/usr/bin/env python3
"""
Migration Example: Old vs New System

This example demonstrates how to migrate from the old hardcoded IntegratedGrowthSystem
to the new composable evolution architecture. It shows side-by-side comparisons
and provides clear migration paths.
"""

import sys
import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import both old and new systems for comparison
from src.structure_net.evolution.integrated_growth_system_v2 import IntegratedGrowthSystem
from src.structure_net.evolution.components import (
    create_standard_evolution_system,
    NetworkContext
)
from src.structure_net.core.network_factory import create_standard_network
from src.structure_net.evolution.advanced_layers import ThresholdConfig, MetricsConfig


def load_mnist_data(batch_size=64):
    """Load MNIST dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def old_way_example():
    """Example using the old IntegratedGrowthSystem (now with composable backend)."""
    print("\n" + "="*60)
    print("🔄 OLD WAY (Backward Compatible)")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = load_mnist_data()
    
    # Create network
    network = create_standard_network(
        architecture=[784, 128, 10],
        sparsity=0.02,
        device=device
    )
    
    # OLD API (still works!)
    config = ThresholdConfig()
    metrics_config = MetricsConfig()
    
    system = IntegratedGrowthSystem(network, config, metrics_config)
    
    # This still works exactly as before, but now uses composable backend
    grown_network = system.grow_network(
        train_loader, 
        test_loader, 
        growth_iterations=2,
        epochs_per_iteration=3
    )
    
    print("✅ Old API completed successfully (using new composable backend)")
    return grown_network


def new_way_example():
    """Example using the new composable evolution system."""
    print("\n" + "="*60)
    print("🚀 NEW WAY (Composable System)")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = load_mnist_data()
    
    # Create network
    network = create_standard_network(
        architecture=[784, 128, 10],
        sparsity=0.02,
        device=device
    )
    
    # NEW API (recommended)
    system = create_standard_evolution_system()
    
    # Create network context
    context = NetworkContext(
        network=network,
        data_loader=train_loader,
        device=device,
        metadata={'val_loader': test_loader}
    )
    
    # Run evolution
    evolved_context = system.evolve_network(context, num_iterations=2)
    
    print("✅ New composable API completed successfully")
    print(f"📊 Performance history: {evolved_context.performance_history}")
    print(f"🔧 Components used: {len(system.get_components())}")
    
    return evolved_context.network


def custom_composable_example():
    """Example creating a custom composable system."""
    print("\n" + "="*60)
    print("🔧 CUSTOM COMPOSABLE SYSTEM")
    print("="*60)
    
    from src.structure_net.evolution.components import (
        ComposableEvolutionSystem,
        StandardExtremaAnalyzer,
        ExtremaGrowthStrategy,
        StandardNetworkTrainer
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = load_mnist_data()
    
    # Create network
    network = create_standard_network(
        architecture=[784, 128, 10],
        sparsity=0.02,
        device=device
    )
    
    # Create custom system with specific components
    system = ComposableEvolutionSystem()
    
    # Add custom-configured components
    extrema_analyzer = StandardExtremaAnalyzer()
    extrema_analyzer.configure({
        'dead_threshold': 0.005,  # More sensitive
        'max_batches': 3
    })
    system.add_component(extrema_analyzer)
    
    extrema_strategy = ExtremaGrowthStrategy()
    extrema_strategy.configure({
        'extrema_threshold': 0.4,  # Lower threshold
        'dead_neuron_threshold': 3
    })
    system.add_component(extrema_strategy)
    
    trainer = StandardNetworkTrainer()
    trainer.configure({'learning_rate': 0.0005})  # Slower learning
    system.add_component(trainer)
    
    # Create context and evolve
    context = NetworkContext(
        network=network,
        data_loader=train_loader,
        device=device,
        metadata={'val_loader': test_loader}
    )
    
    evolved_context = system.evolve_network(context, num_iterations=2)
    
    print("✅ Custom composable system completed successfully")
    print(f"📊 Final performance: {evolved_context.performance_history[-1]:.2%}")
    print(f"🔧 Custom components: {[type(c).__name__ for c in system.get_components()]}")
    
    return evolved_context.network


def migration_comparison():
    """Compare all three approaches."""
    print("\n" + "="*80)
    print("📊 MIGRATION COMPARISON")
    print("="*80)
    
    results = {}
    
    try:
        print("Testing old way...")
        old_network = old_way_example()
        results['old_way'] = "✅ Success (backward compatible)"
    except Exception as e:
        results['old_way'] = f"❌ Failed: {e}"
    
    try:
        print("\nTesting new way...")
        new_network = new_way_example()
        results['new_way'] = "✅ Success (composable)"
    except Exception as e:
        results['new_way'] = f"❌ Failed: {e}"
    
    try:
        print("\nTesting custom way...")
        custom_network = custom_composable_example()
        results['custom_way'] = "✅ Success (custom composable)"
    except Exception as e:
        results['custom_way'] = f"❌ Failed: {e}"
    
    print("\n" + "="*80)
    print("📋 MIGRATION RESULTS")
    print("="*80)
    for approach, result in results.items():
        print(f"{approach:15}: {result}")
    
    print("\n🎯 MIGRATION BENEFITS:")
    print("✅ Backward Compatibility: Existing code works without changes")
    print("✅ Modular Components: Mix and match analyzers and strategies")
    print("✅ Better Configuration: Fine-tune each component individually")
    print("✅ Easier Testing: Test components in isolation")
    print("✅ Future-Proof: Easy to add new components")


def show_migration_guide():
    """Show step-by-step migration guide."""
    print("\n" + "="*80)
    print("📖 STEP-BY-STEP MIGRATION GUIDE")
    print("="*80)
    
    print("""
STEP 1: No Changes Required (Immediate Benefits)
    # Your existing code automatically uses the new backend:
    system = IntegratedGrowthSystem(network, config)
    grown_network = system.grow_network(train_loader, val_loader)
    # ✅ Now uses composable system internally!

STEP 2: Gradual Migration (New Features)
    # Start using composable API for new code:
    from src.structure_net.evolution.components import create_standard_evolution_system
    
    system = create_standard_evolution_system()
    context = NetworkContext(network, train_loader, device)
    evolved_context = system.evolve_network(context, num_iterations=3)

STEP 3: Full Migration (Maximum Benefits)
    # Create custom systems with specific components:
    from src.structure_net.evolution.components import ComposableEvolutionSystem
    
    system = ComposableEvolutionSystem()
    system.add_component(StandardExtremaAnalyzer())
    system.add_component(ExtremaGrowthStrategy())
    # ... configure as needed

STEP 4: Advanced Customization
    # Configure individual components:
    analyzer = StandardExtremaAnalyzer()
    analyzer.configure({'dead_threshold': 0.005})
    system.add_component(analyzer)

MIGRATION TOOLS AVAILABLE:
    • Migration Helper: Analyzes your code and suggests changes
    • Performance Comparison: Tests old vs new performance
    • Documentation: Complete guides and examples
    • Backward Compatibility: Zero breaking changes
""")


if __name__ == "__main__":
    print("🔄 STRUCTURE NET MIGRATION EXAMPLE")
    print("Demonstrating migration from old to new composable system")
    
    # Show migration guide
    show_migration_guide()
    
    # Run comparison
    migration_comparison()
    
    print("\n🎉 Migration example complete!")
    print("Your existing code continues to work while gaining composable benefits.")
