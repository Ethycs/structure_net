#!/usr/bin/env python3
"""
Example demonstrating the refactored adaptive learning rates system.

This example shows how to use the new modular adaptive learning rate system
with different strategies and configurations.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Import the refactored adaptive learning rates system
from src.structure_net.evolution.adaptive_learning_rates import (
    create_adaptive_manager,
    create_adaptive_training_loop,
    create_basic_manager,
    create_advanced_manager,
    create_comprehensive_manager,
    create_ultimate_manager,
    create_preset_manager,
    create_scheduler_presets,
    LearningRateStrategy
)


def create_dummy_network():
    """Create a simple network for demonstration."""
    return nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )


def create_dummy_data():
    """Create dummy data for demonstration."""
    # Generate random data
    X_train = torch.randn(1000, 784)
    y_train = torch.randint(0, 10, (1000,))
    X_val = torch.randn(200, 784)
    y_val = torch.randint(0, 10, (200,))
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    return train_loader, val_loader


def demonstrate_basic_usage():
    """Demonstrate basic usage of the adaptive learning rate system."""
    print("🔧 Basic Usage Example")
    print("=" * 50)
    
    # Create network and data
    network = create_dummy_network()
    train_loader, val_loader = create_dummy_data()
    
    # Create a basic adaptive manager
    manager = create_basic_manager(network, base_lr=0.001)
    
    # Print configuration
    manager.print_rates_summary()
    
    # Create optimizer
    optimizer = manager.create_adaptive_optimizer()
    
    # Simulate a few epochs
    for epoch in range(3):
        manager.update_learning_rates(optimizer, epoch)
        print(f"Epoch {epoch}: Current LRs = {optimizer.get_current_lrs()}")
    
    print()


def demonstrate_strategy_comparison():
    """Demonstrate different strategies."""
    print("🎯 Strategy Comparison")
    print("=" * 50)
    
    network = create_dummy_network()
    strategies = ['basic', 'advanced', 'comprehensive', 'ultimate']
    
    for strategy in strategies:
        print(f"\n📊 Strategy: {strategy}")
        manager = create_adaptive_manager(network, base_lr=0.001, strategy=strategy)
        summary = manager.get_current_rates_summary()
        print(f"   Active schedulers: {len(summary['schedulers'])}")
        for scheduler_name in summary['schedulers'].keys():
            print(f"   - {scheduler_name}")
    
    print()


def demonstrate_preset_configurations():
    """Demonstrate preset configurations."""
    print("⚙️ Preset Configurations")
    print("=" * 50)
    
    network = create_dummy_network()
    presets = create_scheduler_presets()
    
    for preset_name in presets.keys():
        print(f"\n🔧 Preset: {preset_name}")
        try:
            manager = create_preset_manager(network, preset_name, base_lr=0.001)
            summary = manager.get_current_rates_summary()
            print(f"   Strategy: {summary['strategy']}")
            print(f"   Schedulers: {list(summary['schedulers'].keys())}")
        except Exception as e:
            print(f"   Error: {e}")
    
    print()


def demonstrate_custom_configuration():
    """Demonstrate custom scheduler configuration."""
    print("🛠️ Custom Configuration")
    print("=" * 50)
    
    network = create_dummy_network()
    
    # Custom scheduler configurations
    custom_configs = {
        'exponential_backoff': {
            'initial_lr': 0.5,
            'decay_rate': 0.92,
            'min_lr': 1e-5
        },
        'layerwise_rates': {
            'early_rate': 0.005,
            'middle_rate': 0.01,
            'late_rate': 0.002
        }
    }
    
    # Create manager with custom configs
    manager = create_adaptive_manager(
        network=network,
        base_lr=0.001,
        strategy='basic',
        custom_configs=custom_configs
    )
    
    print("Custom configuration applied:")
    manager.print_rates_summary()
    print()


def demonstrate_training_loop():
    """Demonstrate the complete training loop."""
    print("🚀 Complete Training Loop")
    print("=" * 50)
    
    network = create_dummy_network()
    train_loader, val_loader = create_dummy_data()
    
    # Custom progress callback
    def progress_callback(epoch_data):
        if epoch_data['epoch'] % 5 == 0:
            print(f"   Progress: Epoch {epoch_data['epoch']}, "
                  f"Val Acc: {epoch_data['val_acc']:.2%}")
    
    # Run training with adaptive learning rates
    trained_network, history = create_adaptive_training_loop(
        network=network,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=20,
        base_lr=0.001,
        strategy='advanced',
        progress_callback=progress_callback
    )
    
    # Print final results
    final_acc = history[-1]['val_acc']
    print(f"\n✅ Training completed! Final validation accuracy: {final_acc:.2%}")
    
    # Show learning rate evolution
    print("\n📈 Learning Rate Evolution:")
    for i in range(0, len(history), 5):
        epoch_data = history[i]
        lr_summary = epoch_data['learning_rates']
        print(f"   Epoch {i}: Base LR = {lr_summary['base_lr']:.6f}")
    
    print()


def demonstrate_individual_schedulers():
    """Demonstrate individual scheduler usage."""
    print("🔍 Individual Schedulers")
    print("=" * 50)
    
    # Import individual schedulers
    from src.structure_net.evolution.adaptive_learning_rates import (
        ExponentialBackoffScheduler,
        LayerwiseAdaptiveRates,
        SoftClampingScheduler
    )
    
    # Create and test individual schedulers
    print("📉 Exponential Backoff Scheduler:")
    backoff = ExponentialBackoffScheduler(base_lr=0.001, initial_lr=1.0, decay_rate=0.9)
    for epoch in range(5):
        backoff.update_epoch(epoch)
        lr = backoff.get_learning_rate()
        print(f"   Epoch {epoch}: LR = {lr:.6f}")
    
    print("\n🏗️ Layerwise Adaptive Rates:")
    layerwise = LayerwiseAdaptiveRates(base_lr=0.001, total_layers=5)
    for layer_idx in range(5):
        lr = layerwise.get_learning_rate(layer_idx=layer_idx)
        print(f"   Layer {layer_idx}: LR = {lr:.6f}")
    
    print("\n🔒 Soft Clamping Scheduler:")
    clamping = SoftClampingScheduler(base_lr=0.001, max_age=10)
    for age in range(5):
        lr = clamping.get_connection_rate("test_connection", age=age)
        print(f"   Age {age}: LR = {lr:.6f}")
    
    print()


def main():
    """Run all demonstrations."""
    print("🎓 Refactored Adaptive Learning Rates Demo")
    print("=" * 60)
    print()
    
    try:
        demonstrate_basic_usage()
        demonstrate_strategy_comparison()
        demonstrate_preset_configurations()
        demonstrate_custom_configuration()
        demonstrate_individual_schedulers()
        demonstrate_training_loop()
        
        print("🎉 All demonstrations completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
